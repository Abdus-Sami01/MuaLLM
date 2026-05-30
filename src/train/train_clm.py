"""Causal LM pretrain: Predict next token."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_MIN_BODY = 8  # min real tokens in a window (mirrors chunk_text's `< 8: break`)
_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class CLMDataset(Dataset):
    def __init__(self, chunks, vocab_size, pad_id, cls_id, sep_id,
                 special_ids=None, max_len=256):
        self.chunks = chunks
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        body = list(self.chunks[idx])[: self.max_len - 2]
        # For CLM, the sequence is [CLS] <text> [SEP]
        # input is sequence[:-1], target is sequence[1:]
        seq = [self.cls_id] + body + [self.sep_id]
        
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = seq + [self.pad_id] * pad_len
            
        input_ids = seq[:-1]
        labels = seq[1:]
        
        # Mask out padding in labels
        labels = [l if l != self.pad_id else -100 for l in labels]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PackedCLMDataset(Dataset):
    """Windowed CLM over a flat memmapped token stream.

    Reads the `.bin` written by `src.data.pack_tokens` instead of holding token
    chunks in RAM. Each item is a `[CLS] <window> [SEP]` slice, padded to
    `max_len`, with the same `input_ids`/`labels` (`-100` on pad) scheme as
    `CLMDataset`, so the existing train loop consumes it unchanged.

    The memmap is opened lazily so the dataset is safe to fork across DataLoader
    workers (each worker gets its own handle).

    Args:
        bin_path: path to the packed token .bin.
        step: tokens to advance between window starts. Defaults to the body
            length (`max_len - 2`) = non-overlapping windows. Pass a smaller
            value for overlap.
        n_tokens: total tokens in the file; inferred from file size if omitted.
    """

    def __init__(self, bin_path, *, cls_id, sep_id, pad_id, max_len=256,
                 step=None, dtype="uint16", n_tokens=None):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        self.bin_path = str(bin_path)
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.body_len = max_len - 2          # leave room for [CLS] + [SEP]
        self.np_dtype = _DTYPES[dtype]
        self.step = max(1, step if step is not None else self.body_len)

        if n_tokens is None:
            itemsize = np.dtype(self.np_dtype).itemsize
            n_tokens = Path(self.bin_path).stat().st_size // itemsize
        self.n_tokens = int(n_tokens)

        # window starts at 0, step, 2*step, ... while >= _MIN_BODY tokens remain
        last_start = self.n_tokens - _MIN_BODY
        self.n_windows = 0 if last_start < 0 else (last_start // self.step) + 1

        self._data = None  # lazy per-worker memmap

    def _mm(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")
        return self._data

    def close(self):
        """Release the memmap handle (Windows locks the file while open)."""
        if self._data is not None:
            mm = getattr(self._data, "_mmap", None)
            if mm is not None:
                mm.close()
            self._data = None

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.n_windows
        if not 0 <= idx < self.n_windows:
            raise IndexError(idx)
        data = self._mm()
        start = idx * self.step
        body = data[start:start + self.body_len].tolist()
        seq = [self.cls_id] + body + [self.sep_id]

        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = seq + [self.pad_id] * pad_len

        input_ids = seq[:-1]
        labels = seq[1:]
        labels = [l if l != self.pad_id else -100 for l in labels]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_clm(decoder, clm_head, dataset, *, epochs=1, batch_size=8,
              lr=1e-4, weight_decay=0.01, grad_clip=1.0, device="cpu",
              log_every=20, num_workers=0):
    """Train decoder + CLM head jointly. Returns list of per-step losses."""
    decoder.train()
    clm_head.train()
    decoder.to(device)
    clm_head.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)
    
    seen, params = set(), []
    for p in list(decoder.parameters()) + list(clm_head.parameters()):
        if id(p) in seen:
            continue
        seen.add(id(p))
        params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    losses = []
    step = 0
    for ep in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Autoregressive forward pass
            hidden = decoder(input_ids)
            logits = clm_head(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"[ep {ep} step {step}] loss={loss.item():.4f}")
            step += 1
    return losses


def save_checkpoint(path, decoder, clm_head=None, opt=None, meta=None):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"decoder": decoder.state_dict(), "meta": meta or {}}
    if clm_head is not None:
        payload["clm_head"] = clm_head.state_dict()
    if opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, path)
