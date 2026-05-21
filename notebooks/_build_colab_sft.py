"""Generate notebooks/colab_sft.ipynb - GPU fine-tuning notebook.

Run once:  python notebooks/_build_colab_sft.py

The notebook is self-contained: loads YOUR pretrained checkpoint + tokenizer +
QA jsonl files (uploaded by you), runs SFT on GPU, downloads the chatbot.
Model code matches the local repo exactly (causal decoder).
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "colab_sft.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


cells = []

cells.append(md(r"""# slm_qa - Supervised Fine-Tuning (SFT) on GPU

Fine-tunes your pretrained causal LM into a teaching-QA chatbot.

**What you upload** (cell 9):
- `pretrain_softmax_*.pt`  - the checkpoint from the pretrain notebook
- `tokenizer.json`         - the matching tokenizer (vocab 8000)
- `pk_teaching_qa.jsonl` + `github_edu_qa.jsonl` - your QA data

**Runtime**: set to GPU (Colab T4 / Kaggle T4). 10 epochs over ~2060
dialogues finishes in a few minutes on GPU (vs ~50 hrs on CPU).

Model code below is identical to the local repo (causal decoder, fixed
tokenization). Run cells top to bottom.

---
"""))

cells.append(md("## 1. Environment check"))

cells.append(code(r"""import sys, platform, time, math, random
import torch
print('python :', sys.version.split()[0])
print('torch  :', torch.__version__)
print('cuda   :', torch.cuda.is_available())
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print('device :', d.name, '|', round(d.total_memory / 1e9, 1), 'GB')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE :', DEVICE)
"""))

cells.append(code(r"""import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers'])
"""))

cells.append(md(r"""## 2. Model code (inline, causal decoder)

Three attention variants + decoder + causal LM head. Identical to the
local `src/model/` package.
"""))

cells.append(code(r'''"""Softmax attention - causal (decoder). O(N^2 D)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
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

        # Causal mask: position t attends only to <= t
        causal_mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

        if attn_mask is not None:
            keep = attn_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~keep, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
'''))

cells.append(code(r'''"""Linear attention - causal cumulative form. O(N D^2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x):
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
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

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        if attn_mask is not None:
            m = attn_mask[:, None, :, None].to(q.dtype)
            k = k * m
            v = v * m

        # causal: prefix sums so position t sees only <= t
        k_cumsum = torch.cumsum(k, dim=2)
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)

        num = torch.einsum('bhnd,bhnde->bhne', q, kv_cumsum)
        denom = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum).clamp(min=self.eps)
        out = num / denom.unsqueeze(-1)

        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
'''))

cells.append(code(r'''"""RWKV-style time-mix - causal (forward-only). O(N D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wkv_forward(k, v, w_log_decay, u_bonus):
    B, H, N, D = k.shape
    out = torch.zeros_like(v)
    NEG_INF = torch.full((B, H, D), -1e30, device=k.device, dtype=k.dtype)
    num = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    den = torch.zeros(B, H, D, device=k.device, dtype=k.dtype)
    max_log = NEG_INF.clone()
    w_log = w_log_decay
    for t in range(N):
        kt = k[:, :, t]
        vt = v[:, :, t]
        kt_b = kt + u_bonus
        new_max = torch.maximum(max_log + w_log, kt_b)
        e1 = torch.exp(max_log + w_log - new_max)
        e2 = torch.exp(kt_b - new_max)
        out_num = e1 * num + e2 * vt
        out_den = e1 * den + e2
        out[:, :, t] = out_num / (out_den + 1e-6)
        new_max_state = torch.maximum(max_log + w_log, kt)
        es1 = torch.exp(max_log + w_log - new_max_state)
        es2 = torch.exp(kt - new_max_state)
        num = es1 * num + es2 * vt
        den = es1 * den + es2
        max_log = new_max_state
    return out


class RWKVTimeMix(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bidirectional=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
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
        out_b = torch.sigmoid(r_b) * wkv_b
        out = torch.cat([out_f, out_b], dim=-1).transpose(1, 2).contiguous().view(B, N, 2 * D)
        return self.out_proj(self.dropout(out))
'''))

cells.append(code(r'''"""Decoder block, decoder stack, causal LM head."""
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


class DecoderBlock(nn.Module):
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


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, max_len=512, attention="softmax",
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_len, dropout, pad_id)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, attention, dropout)
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


class CausalLMHead(nn.Module):
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


print("model defined")
'''))

cells.append(md(r"""## 3. Upload your files

Run the cell, then pick **all** of these from your computer:
- the pretrain checkpoint `.pt`
- `tokenizer.json`
- the QA `.jsonl` files

(Big checkpoint slow to upload? Mount Google Drive instead - see the
commented block in the next cell.)
"""))

cells.append(code(r'''from google.colab import files
print("Select: pretrain .pt  +  tokenizer.json  +  *.jsonl files")
uploaded = files.upload()
print()
print("uploaded:", list(uploaded.keys()))

# --- Google Drive alternative (uncomment if upload is too slow) ---
# from google.colab import drive
# drive.mount("/content/drive")
# DRIVE_DIR = "/content/drive/MyDrive/slm_qa"   # put your files here
# import os
# uploaded = {f: None for f in os.listdir(DRIVE_DIR)}
# os.chdir(DRIVE_DIR)
'''))

cells.append(code(r'''"""Locate uploaded files, load tokenizer + checkpoint, rebuild the model."""
from tokenizers import Tokenizer

ckpt_files  = sorted(f for f in uploaded if f.endswith(".pt"))
tok_files   = sorted(f for f in uploaded if f.endswith(".json"))
jsonl_files = sorted(f for f in uploaded if f.endswith(".jsonl"))

assert ckpt_files,  "no .pt checkpoint uploaded"
assert tok_files,   "no tokenizer .json uploaded"
assert jsonl_files, "no .jsonl QA files uploaded"

CKPT_PATH  = ckpt_files[0]
TOK_PATH   = tok_files[0]
DATA_PATHS = jsonl_files
print("checkpoint :", CKPT_PATH)
print("tokenizer  :", TOK_PATH)
print("data       :", DATA_PATHS)

tok = Tokenizer.from_file(TOK_PATH)
ck = torch.load(CKPT_PATH, map_location=DEVICE)
config = ck["config"]
print("config:", config)

assert tok.get_vocab_size() == config["vocab_size"], (
    f"vocab mismatch: tokenizer={tok.get_vocab_size()} "
    f"checkpoint={config['vocab_size']} - upload the matching tokenizer")

decoder = Decoder(
    vocab_size=config["vocab_size"], d_model=config["d_model"],
    n_heads=config["n_heads"], n_layers=config["n_layers"],
    d_ff=config["d_ff"], max_len=config["max_len"],
    attention=config["attention"], pad_id=config["pad_id"],
).to(DEVICE)
decoder.load_state_dict(ck["decoder"])

clm = CausalLMHead(config["d_model"], config["vocab_size"],
                   tied_weight=decoder.embed.tok.weight).to(DEVICE)
clm.load_state_dict(ck["clm_head"])

n_params = sum(p.numel() for p in decoder.parameters())
print(f"model loaded. params={n_params:,}  attention={config['attention']}")
'''))

cells.append(md(r"""## 4. SFT dataset + training loop

`SFTDataset` encodes each dialogue in one pass (`add_special_tokens=False`)
- no per-word `[CLS]/[SEP]` corruption.
"""))

cells.append(code(r'''"""SFT dataset: next-token prediction over chat dialogues."""
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SFTDataset(Dataset):
    def __init__(self, jsonl_paths, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.samples.append(json.loads(line)["text"])
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        print(f"SFTDataset: {len(self.samples)} samples from {len(jsonl_paths)} file(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # whole-text encode; post-processor template NOT re-applied
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [self.cls_id] + enc.ids
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        input_ids = ids[:-1]
        labels = ids[1:]
        labels = [l if l != self.pad_id else -100 for l in labels]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
'''))

cells.append(code(r'''"""SFT training loop: AMP + LR warmup + cosine decay."""
from torch.cuda.amp import autocast, GradScaler


def pick_amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32, False
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    return torch.float16, True


def train_sft(decoder, clm_head, dataset, *, epochs=10, batch_size=8,
              lr=3e-5, weight_decay=0.01, grad_clip=1.0, warmup_steps=100,
              device="cuda", log_every=20):
    decoder.train(); clm_head.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    seen, params = set(), []
    for p in list(decoder.parameters()) + list(clm_head.parameters()):
        if id(p) in seen:
            continue
        seen.add(id(p)); params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    total = max(1, len(loader) * epochs)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        prog = (s - warmup_steps) / max(1, total - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp_dtype, use_amp = pick_amp_dtype()
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)
    print(f"AMP dtype={amp_dtype}  total_steps={total}  warmup={warmup_steps}")

    losses = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                hidden = decoder(input_ids)
                logits = clm_head(hidden)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       labels.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            sched.step()
            losses.append(loss.item())
            if step % log_every == 0:
                dt = time.time() - t0
                print(f"[ep {ep} step {step:5d}] loss={loss.item():.4f}  "
                      f"ppl={math.exp(min(loss.item(), 20)):.1f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  elapsed={dt:.1f}s")
            step += 1
    return losses
'''))

cells.append(md(r"""## 5. Run fine-tuning

10 epochs over ~2060 dialogues. On a T4 this is a few minutes.
"""))

cells.append(code(r'''# ===== SFT config =====
EPOCHS       = 10
BATCH_SIZE   = 8
LR           = 3e-5      # low - we are fine-tuning, not pretraining
WARMUP_STEPS = 100
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)

ds = SFTDataset(DATA_PATHS, tok, max_len=config["max_len"])
losses = train_sft(
    decoder, clm, ds,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    warmup_steps=WARMUP_STEPS, device=str(DEVICE),
)
tail = min(50, len(losses))
print(f"\nfinal loss (last {tail}): {sum(losses[-tail:]) / tail:.4f}")
'''))

cells.append(md("## 6. Save + download the fine-tuned chatbot"))

cells.append(code(r'''out_path = f"chatbot_finetuned_{int(time.time())}.pt"
torch.save({
    "decoder": decoder.state_dict(),
    "clm_head": clm.state_dict(),
    "config": config,
    "tokenizer_path": TOK_PATH,
    "meta": {"finetuned": True, "epochs": EPOCHS, "config": config,
             "tokenizer_path": TOK_PATH},
}, out_path)
print(f"saved: {out_path}  ({__import__('os').path.getsize(out_path)/1e6:.1f} MB)")

try:
    from google.colab import files
    files.download(out_path)
except ImportError:
    print("not on Colab - file is in the working dir")
'''))

cells.append(md(r"""## 7. Test the chatbot

Generates an answer token-by-token. Stops at `[SEP]`.

Reminder: this is a generative model trained from scratch on small data.
Expect fluent but often factually-wrong answers. It is not a knowledge base.
"""))

cells.append(code(r'''@torch.no_grad()
def generate(prompt, max_new_tokens=120, temperature=0.8, top_k=40):
    decoder.eval(); clm.eval()
    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    max_len = config["max_len"]
    ids = [cls_id] + tok.encode(prompt, add_special_tokens=False).ids
    gen = []
    for _ in range(max_new_tokens):
        inp = torch.tensor([ids[-max_len:]], dtype=torch.long, device=DEVICE)
        logits = clm(decoder(inp))[0, -1, :] / max(temperature, 1e-5)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == sep_id:
            break
        ids.append(nxt)
        gen.append(nxt)
    return tok.decode(gen)


def ask(question):
    out = generate(f"User: {question}\nBot:")
    print(f"User: {question}")
    print(f"Bot: {out}\n")


ask("What is the Single National Curriculum in Pakistan?")
ask("How can I manage a large classroom?")
ask("What is the difference between formative and summative assessment?")
'''))

cells.append(md(r"""## 8. Next steps

- Download `chatbot_finetuned_*.pt`. Use it locally:
  `python -m src.infer.generate --ckpt chatbot_finetuned_xxx.pt`
- Answers fluent but unreliable? That is expected for a from-scratch
  generative LM. For factually-correct answers, add a retrieval (RAG)
  layer that feeds the relevant guide passage into the prompt.
- For the attention ablation: re-run the pretrain notebook with
  `ATTENTION = "linear"` / `"rwkv"`, then SFT each here.
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}  ({NB_PATH.stat().st_size / 1024:.1f} KB)  cells={len(cells)}")
