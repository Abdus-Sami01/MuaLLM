"""Causal LM pretrain: Predict next token."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
