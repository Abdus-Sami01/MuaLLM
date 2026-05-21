"""Generate notebooks/colab_pretrain.ipynb from inline cell sources.

Run once:  python notebooks/_build_colab_nb.py
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "colab_pretrain.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = []

cells.append(md(r"""# slm_qa - Pretrain on GPU (Colab / Kaggle)

Trains a small encoder language model with one of three custom attention variants
on a Wikipedia education subset, then saves a checkpoint you can download and
use locally for extractive QA fine-tuning.

**Variants** (swappable):
- `softmax` - vanilla scaled dot-product, O(N^2 D)
- `linear`  - Katharopoulos linear attention, O(N D^2)
- `rwkv`    - bidirectional RWKV-style time-mix, O(N D)

**Runtime**: set to GPU (Colab T4 or Kaggle T4 / P100). Whole pretrain
finishes in minutes, not hours.

**Self-contained**: every line of model + training code lives in this notebook.
No GitHub clone needed.

---
"""))

cells.append(md("## 1. Environment check"))

cells.append(code(r"""import sys, os, platform
print('python   :', sys.version.split()[0])
print('platform :', platform.platform())

import torch
print('torch    :', torch.__version__)
print('cuda     :', torch.cuda.is_available())
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print('device   :', d.name)
    print('vram GB  :', round(d.total_memory / 1e9, 1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE   :', DEVICE)
"""))

cells.append(code(r"""# install missing deps (torch is preinstalled with CUDA on Colab/Kaggle)
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'tokenizers', 'datasets'])
"""))

cells.append(md(r"""## 2. Model code (inline)

Three attention variants share an encoder skeleton. Swap by name.
"""))

cells.append(code(r'''"""Softmax attention - baseline, O(N^2 D)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not div by n_heads={n_heads}")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            keep = attn_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~keep, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
'''))

cells.append(code(r'''"""Linear attention - Katharopoulos 2020, O(N D^2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x):
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not div by n_heads={n_heads}")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = elu_feature_map(q); k = elu_feature_map(k)
        if attn_mask is not None:
            m = attn_mask[:, None, :, None].to(q.dtype)
            k = k * m; v = v * m
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        num = torch.einsum("bhnd,bhde->bhne", q, kv)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp(min=self.eps)
        out = num / denom.unsqueeze(-1)
        out = self.dropout(out).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
'''))

cells.append(code(r'''"""RWKV-style time-mix - bidirectional, O(N D). GPU note: Python loop over
sequence length is unavoidable in pure PyTorch; for short N (<=512) it is
acceptable. Real RWKV uses fused CUDA kernels for speed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wkv_forward(k, v, w_log_decay, u_bonus):
    B, H, N, D = k.shape
    out = torch.zeros_like(v)
    num = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    den = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    max_log = torch.full((B, H, D), -1e30, device=k.device, dtype=k.dtype)
    w_log = w_log_decay
    for t in range(N):
        kt = k[:, :, t]; vt = v[:, :, t]
        kt_b = kt + u_bonus
        nm = torch.maximum(max_log + w_log, kt_b)
        e1 = torch.exp(max_log + w_log - nm)
        e2 = torch.exp(kt_b - nm)
        on = e1 * num + e2 * vt
        od = e1 * den + e2
        out[:, :, t] = on / (od + 1e-6)
        nms = torch.maximum(max_log + w_log, kt)
        es1 = torch.exp(max_log + w_log - nms)
        es2 = torch.exp(kt - nms)
        num = es1 * num + es2 * vt
        den = es1 * den + es2
        max_log = nms
    return out


class RWKVTimeMix(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bidirectional=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not div by n_heads={n_heads}")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.bidirectional = bidirectional
        self.mix_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mix_r = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.time_decay_raw = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.time_bonus = nn.Parameter(torch.zeros(n_heads, self.d_head))
        out_dim = 2 * d_model if bidirectional else d_model
        self.out_proj = nn.Linear(out_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _shift_right(x):
        return F.pad(x, (0, 0, 1, 0))[:, :-1]

    def _project(self, x_orig, shifted):
        xk = x_orig * self.mix_k + shifted * (1 - self.mix_k)
        xv = x_orig * self.mix_v + shifted * (1 - self.mix_v)
        xr = x_orig * self.mix_r + shifted * (1 - self.mix_r)
        B, N, _ = x_orig.shape
        k = self.key(xk).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.value(xv).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        r = self.receptance(xr).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        return k, v, r

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        w_log = -torch.exp(self.time_decay_raw)
        k_f, v_f, r_f = self._project(x, self._shift_right(x))
        if attn_mask is not None:
            m = attn_mask[:, None, :, None].to(k_f.dtype)
            k_f = k_f.masked_fill(m == 0, -1e9)
            v_f = v_f * m
        wkv_f = _wkv_forward(k_f, v_f, w_log, self.time_bonus)
        out_f = torch.sigmoid(r_f) * wkv_f
        if not self.bidirectional:
            out = out_f.transpose(1, 2).contiguous().view(B, N, D)
            return self.out_proj(self.dropout(out))
        x_rev = x.flip(dims=[1])
        k_b, v_b, r_b = self._project(x_rev, self._shift_right(x_rev))
        if attn_mask is not None:
            m_rev = attn_mask.flip(dims=[1])[:, None, :, None].to(k_b.dtype)
            k_b = k_b.masked_fill(m_rev == 0, -1e9)
            v_b = v_b * m_rev
        wkv_b = _wkv_forward(k_b, v_b, w_log, self.time_bonus).flip(dims=[2])
        out_b = torch.sigmoid(r_b.flip(dims=[2])) * wkv_b
        out = torch.cat([out_f, out_b], dim=-1).transpose(1, 2).contiguous().view(B, N, 2 * D)
        return self.out_proj(self.dropout(out))
'''))

cells.append(code(r'''"""Encoder block, encoder stack, task heads."""
import torch
import torch.nn as nn


ATTENTION_REGISTRY = {
    "softmax": SoftmaxAttention,
    "linear":  LinearAttention,
    "rwkv":    RWKVTimeMix,
}


def build_attention(name, d_model, n_heads, dropout=0.0):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"unknown attention {name!r}. options={list(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[name](d_model, n_heads, dropout)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attention="softmax", dropout=0.0):
        super().__init__()
        self.attn = build_attention(attention, d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input_ids):
        B, N = input_ids.shape
        if N > self.max_len:
            raise ValueError(f"seq {N} > max_len {self.max_len}")
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok(input_ids) + self.pos(positions)
        return self.dropout(self.ln(x))


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, max_len=512, attention="softmax",
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_len, dropout, pad_id)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, attention, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.pad_id = pad_id
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.attention_name = attention

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        return self.ln_f(x)


class MLMHead(nn.Module):
    def __init__(self, d_model, vocab_size, tied_weight=None):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        if tied_weight is not None:
            self.decoder.weight = tied_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden):
        h = self.ln(self.act(self.transform(hidden)))
        return self.decoder(h) + self.bias


class QASpanHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 2)

    def forward(self, hidden, attention_mask=None):
        logits = self.proj(hidden)
        start, end = logits.split(1, dim=-1)
        start = start.squeeze(-1); end = end.squeeze(-1)
        if attention_mask is not None:
            mask = (attention_mask == 0)
            start = start.masked_fill(mask, -1e9)
            end = end.masked_fill(mask, -1e9)
        return start, end


print("model defined")
'''))

cells.append(md(r"""## 3. Data pipeline

Stream Wikipedia, filter to education-related articles, save plain text,
train BPE on it, chunk into token windows.
"""))

cells.append(code(r'''"""Download a wiki education subset (one-time, ~50 MB by default)."""
from pathlib import Path
from datasets import load_dataset


KEYWORDS = [
    "teacher", "teaching", "education", "pedagogy", "classroom",
    "curriculum", "lesson plan", "school", "learning", "assessment",
    "instruction", "literacy", "numeracy", "kindergarten",
    "tutor", "professor", "lecturer", "training", "course",
    "university", "college", "academy", "student", "pupil",
    "literacy", "scholar", "didactic", "syllabus",
]


def download_subset(out_path, max_bytes=100 * 1024 * 1024, lang="en", max_articles=30000):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True,
    )
    written, n = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            title = (row.get("title") or "").lower()
            text = row.get("text") or ""
            if not text or not any(k in title for k in KEYWORDS):
                continue
            f.write(text.strip() + "\n\n")
            written += len(text.encode("utf-8"))
            n += 1
            if written >= max_bytes or n >= max_articles:
                break
    return {"bytes": written, "articles": n, "path": str(out)}


DATA_DIR = Path("data")
RAW = DATA_DIR / "raw" / "wiki_edu.txt"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    print("downloading (~100 MB) ...")
    stats = download_subset(RAW, max_bytes=100 * 1024 * 1024, max_articles=30000)
    print(stats)
else:
    print(f"already downloaded: {RAW} ({RAW.stat().st_size / 1024 / 1024:.1f} MB)")
    print("delete the file to redownload with the wider keyword set.")
'''))

cells.append(code(r'''"""Train byte-level BPE tokenizer with QA special tokens."""
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_bpe(text_files, output_path, vocab_size=8000, min_frequency=2):
    text_files = [str(p) for p in text_files]
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t.train(text_files, trainer)
    cls_id, sep_id = t.token_to_id("[CLS]"), t.token_to_id("[SEP]")
    t.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    t.save(str(output_path))
    return t


TOK_PATH = DATA_DIR / "processed" / "tokenizer.json"
if not TOK_PATH.exists():
    print("training BPE ...")
    tok = train_bpe([RAW], TOK_PATH, vocab_size=8000, min_frequency=2)
    print(f"vocab_size={tok.get_vocab_size()}  saved -> {TOK_PATH}")
else:
    tok = Tokenizer.from_file(str(TOK_PATH))
    print(f"loaded tokenizer  vocab_size={tok.get_vocab_size()}")
'''))

cells.append(code(r'''"""Stream-tokenize corpus into windows.
Avoids OOM: encodes in paragraph batches, never holds whole-corpus encoding.
"""
from tqdm.auto import tqdm


def stream_chunks(path, tokenizer, max_tokens=254, stride=64,
                  batch_paragraphs=512, min_chunk=8):
    out = []
    buf_ids = []
    step = max(1, max_tokens - stride)
    para_buf = []

    def flush(paras):
        if not paras:
            return []
        encs = tokenizer.encode_batch(paras, add_special_tokens=False)
        ids = []
        for e in encs:
            ids.extend(e.ids)
        return ids

    def window(ids_list):
        windows = []
        i = 0
        while i + max_tokens <= len(ids_list):
            windows.append(ids_list[i:i + max_tokens])
            i += step
        # keep tail for next batch
        return windows, ids_list[i:]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="chunking", unit="lines"):
            line = line.strip()
            if not line:
                if para_buf:
                    buf_ids.extend(flush(para_buf))
                    para_buf = []
                    wins, buf_ids = window(buf_ids)
                    out.extend(wins)
                continue
            para_buf.append(line)
            if len(para_buf) >= batch_paragraphs:
                buf_ids.extend(flush(para_buf))
                para_buf = []
                wins, buf_ids = window(buf_ids)
                out.extend(wins)
    # final flush
    if para_buf:
        buf_ids.extend(flush(para_buf))
    if len(buf_ids) >= min_chunk:
        out.append(buf_ids[:max_tokens])
    return out


chunks = stream_chunks(RAW, tok, max_tokens=254, stride=64)
print(f"chunks: {len(chunks):,}  approx tokens: {sum(len(c) for c in chunks):,}")
'''))

cells.append(md(r"""## 4. MLM dataset + training loop

Mixed-precision (AMP) for GPU speed. AdamW + grad clip. Saves a checkpoint
you can download.
"""))

cells.append(code(r'''"""MLM dataset: 15% mask, 80/10/10 (mask / random / keep)."""
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MLMDataset(Dataset):
    def __init__(self, chunks, vocab_size, mask_id, pad_id, cls_id, sep_id,
                 special_ids=None, mask_prob=0.15, max_len=256):
        self.chunks = chunks
        self.vocab_size = vocab_size
        self.mask_id, self.pad_id = mask_id, pad_id
        self.cls_id, self.sep_id = cls_id, sep_id
        self.special_ids = set(special_ids or [])
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        body = list(self.chunks[idx])[: self.max_len - 2]
        ids = [self.cls_id] + body + [self.sep_id]
        labels = [-100] * len(ids)
        input_ids = list(ids)
        for i, tok in enumerate(ids):
            if tok in self.special_ids:
                continue
            if random.random() < self.mask_prob:
                labels[i] = tok
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = random.randrange(self.vocab_size)
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
'''))

cells.append(code(r'''"""Training loop with mixed precision, LR warmup + cosine decay, grad accum."""
import time, math
from torch.cuda.amp import autocast, GradScaler


def pick_amp_dtype():
    """Prefer bf16 on Ampere+ (more stable for tiny losses); else fp16; else fp32."""
    if not torch.cuda.is_available():
        return torch.float32, False
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    return torch.float16, True


def make_lr_scheduler(opt, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_mlm_amp(encoder, mlm_head, dataset, *, epochs=1, batch_size=32,
                  lr=3e-4, weight_decay=0.01, grad_clip=1.0,
                  device="cuda", log_every=20, num_workers=2,
                  max_steps=None, warmup_steps=500, grad_accum=1,
                  min_lr_ratio=0.1):
    encoder.train(); mlm_head.train()
    encoder.to(device); mlm_head.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        drop_last=True)
    # dedupe tied params
    seen, params = set(), []
    for p in list(encoder.parameters()) + list(mlm_head.parameters()):
        if id(p) in seen: continue
        seen.add(id(p)); params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    # Estimate total optimizer steps so cosine schedule has the right horizon.
    steps_per_epoch = len(loader) // grad_accum
    total_opt_steps = max_steps if max_steps else steps_per_epoch * epochs
    sched = make_lr_scheduler(opt, warmup_steps, total_opt_steps, min_lr_ratio)

    amp_dtype, use_amp = pick_amp_dtype()
    use_scaler = use_amp and amp_dtype == torch.float16  # bf16 needs no scaler
    scaler = GradScaler(enabled=use_scaler)
    print(f"AMP: dtype={amp_dtype}  scaler={use_scaler}  "
          f"total_opt_steps={total_opt_steps}  warmup={warmup_steps}")

    losses = []
    step = 0; opt_step = 0; t0 = time.time()
    opt.zero_grad()
    for ep in range(epochs):
        for micro_i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                hidden = encoder(input_ids)
                logits = mlm_head(hidden)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(loss.item() * grad_accum)

            do_step = ((micro_i + 1) % grad_accum == 0)
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    opt.step()
                sched.step()
                opt.zero_grad()
                opt_step += 1

                if opt_step % log_every == 0:
                    dt = time.time() - t0
                    eff_bs = batch_size * grad_accum
                    tps = (opt_step * eff_bs * dataset.max_len) / max(dt, 1e-3)
                    cur_lr = sched.get_last_lr()[0]
                    cur_loss = sum(losses[-grad_accum:]) / grad_accum
                    print(f"[ep {ep} opt_step {opt_step:5d}] "
                          f"loss={cur_loss:.4f}  ppl={math.exp(min(cur_loss, 20)):.1f}  "
                          f"lr={cur_lr:.2e}  tok/s={tps:.0f}  elapsed={dt:.1f}s")

                if max_steps and opt_step >= max_steps:
                    return losses
            step += 1
    return losses
'''))

cells.append(md(r"""## 5. Build model + train

Defaults below produce a ~35 M-param model that uses ~8-10 GB on a T4.
If you OOM, halve `BATCH_SIZE` and set `GRAD_ACCUM = 2` to keep the same
effective batch.

Edit `ATTENTION` to pick `softmax`, `linear`, or `rwkv`.
"""))

cells.append(code(r'''# ===== config (tuned for ~10 GB GPU usage on T4 / 16 GB) =====
ATTENTION    = "softmax"   # "softmax" | "linear" | "rwkv"
D_MODEL      = 512
N_HEADS      = 8
N_LAYERS     = 8
D_FF         = 2048
MAX_LEN      = 256
BATCH_SIZE   = 128
GRAD_ACCUM   = 1           # effective batch = BATCH_SIZE * GRAD_ACCUM
LR           = 5e-4
WARMUP_STEPS = 500
MIN_LR_RATIO = 0.1         # cosine floor (10% of peak)
EPOCHS       = 3
MAX_STEPS    = None        # None = full epoch loop. set int to cap.
LOG_EVERY    = 20
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)

# Sanity: corpus must be large enough to actually train a 35M model.
corpus_mb = RAW.stat().st_size / 1024 / 1024
print(f"corpus size: {corpus_mb:.1f} MB  chunks: {len(chunks):,}")
if corpus_mb < 30:
    print(f"WARNING: corpus only {corpus_mb:.1f} MB.")
    print("Loss will plateau near unigram floor. Rerun the download cell with")
    print("max_articles=20000 and broader KEYWORDS, or set max_bytes higher.")

vocab_size = tok.get_vocab_size()
pad_id  = tok.token_to_id("[PAD]")
cls_id  = tok.token_to_id("[CLS]")
sep_id  = tok.token_to_id("[SEP]")
mask_id = tok.token_to_id("[MASK]")
special_ids = [tok.token_to_id(t) for t in SPECIAL_TOKENS]

ds = MLMDataset(chunks, vocab_size, mask_id, pad_id, cls_id, sep_id,
                special_ids=special_ids, max_len=MAX_LEN)
print(f"dataset: {len(ds):,} chunks  max_len={MAX_LEN}")

encoder = Encoder(
    vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_LEN,
    attention=ATTENTION, dropout=0.1, pad_id=pad_id,
)
mlm = MLMHead(D_MODEL, vocab_size, tied_weight=encoder.embed.tok.weight)
n_enc = sum(p.numel() for p in encoder.parameters())
n_mlm = sum(p.numel() for p in mlm.parameters() if id(p) != id(encoder.embed.tok.weight))
print(f"params: encoder={n_enc:,}  mlm_extra={n_mlm:,}  total~={n_enc + n_mlm:,}")
print(f"attention: {ATTENTION}  effective_batch={BATCH_SIZE * GRAD_ACCUM}")
'''))

# Debug: inspect a sample batch BEFORE training so masking issues surface early.
cells.append(md(r"""### Debug: inspect a batch

Verifies the dataset actually masks tokens. If `masked` count is 0 or
`labels != -100` count is 0, training will plateau at the unigram floor.
"""))

cells.append(code(r'''sample = ds[0]
ids = sample["input_ids"].tolist()
lbl = sample["labels"].tolist()
n_total  = len(ids)
n_pad    = sum(1 for t in ids if t == pad_id)
n_masked = sum(1 for t in ids if t == mask_id)
n_labels = sum(1 for x in lbl if x != -100)
print(f"sample len     : {n_total}")
print(f"  pad positions: {n_pad}")
print(f"  [MASK] tokens: {n_masked}")
print(f"  label positions (non -100): {n_labels}")
print(f"  effective mask rate on body: "
      f"{n_labels / max(1, n_total - n_pad - 2):.2%}  (target ~15%)")
print()
print("first 40 ids:", ids[:40])
print("decoded     :", tok.decode([i for i in ids[:40] if i != pad_id]))
assert n_labels > 0, "no labels set - masking broken"
'''))

cells.append(code(r'''losses = train_mlm_amp(
    encoder, mlm, ds,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    device=str(DEVICE), log_every=LOG_EVERY,
    num_workers=2, max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS, grad_accum=GRAD_ACCUM,
    min_lr_ratio=MIN_LR_RATIO,
)
window = min(50, len(losses))
print(f"\nfinal loss (last {window}): {sum(losses[-window:]) / window:.4f}")
print(f"final ppl  : {math.exp(min(sum(losses[-window:]) / window, 20)):.1f}")
'''))

cells.append(md(r"""## 6. Save + download checkpoint

Saves encoder weights (and MLM head) so you can load them locally for
QA fine-tuning. On Colab this triggers a browser download.
"""))

cells.append(code(r'''import time as _time
ckpt_path = Path(f"checkpoints/pretrain_{ATTENTION}_{int(_time.time())}.pt")
ckpt_path.parent.mkdir(parents=True, exist_ok=True)

config = {
    "attention": ATTENTION,
    "vocab_size": vocab_size,
    "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
    "d_ff": D_FF, "max_len": MAX_LEN, "pad_id": pad_id,
}

torch.save({
    "encoder": encoder.state_dict(),
    "mlm_head": mlm.state_dict(),
    "config": config,
    "tokenizer_path": str(TOK_PATH),
    "losses_tail": losses[-100:] if losses else [],
}, ckpt_path)
print(f"saved: {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
'''))

cells.append(code(r'''# Trigger browser download (Colab) or just print the working-dir path (Kaggle).
try:
    from google.colab import files  # Colab
    files.download(str(ckpt_path))
    files.download(str(TOK_PATH))
except ImportError:
    print("Not on Colab. Files saved to:")
    print(f"  checkpoint: {ckpt_path.absolute()}")
    print(f"  tokenizer : {TOK_PATH.absolute()}")
    print("On Kaggle, anything under /kaggle/working/ is downloadable from the sidebar.")
'''))

cells.append(md(r"""## 7. Quick sanity check (optional)

Mask one token in a short sentence and see what the model predicts.
"""))

cells.append(code(r'''@torch.no_grad()
def predict_mask(text, top_k=5):
    enc = tok.encode(text)
    ids = enc.ids
    if mask_id not in ids:
        print("no [MASK] token in input. Insert one manually.")
        return
    ids_t = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    encoder.eval(); mlm.eval()
    hidden = encoder(ids_t)
    logits = mlm(hidden)
    pos = ids.index(mask_id)
    top = torch.topk(logits[0, pos], top_k)
    print(f"input: {text}")
    print(f"top {top_k}:")
    for s, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tok.decode([i])!r}  score={s:.2f}")


# Example - replace [MASK] manually
predict_mask("Teachers play a fundamental role in [MASK] young minds.")
'''))

cells.append(md(r"""## 8. Next steps

1. **Download** `checkpoints/pretrain_<variant>_<timestamp>.pt` and the
   matching `tokenizer.json` to your local machine.
2. **Locally** load them, attach a `QASpanHead`, and fine-tune on your
   synthetic + hand-written QA pairs (extractive span head).
3. **Repeat** this notebook twice more with `ATTENTION = "linear"` and
   `"rwkv"` to get three checkpoints for ablation.

To load locally:

```python
import torch
ck = torch.load("pretrain_softmax_xxx.pt", map_location="cpu")
encoder = Encoder(**ck["config"])
encoder.load_state_dict(ck["encoder"])
# attach QASpanHead, fine-tune on QA pairs
```
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}  ({NB_PATH.stat().st_size / 1024:.1f} KB)  cells={len(cells)}")
