"""Knowledge distillation: student Decoder learns from a teacher LM (Phi-3-mini
by default).

Two subcommands:

  cache  - run the teacher over a corpus once, store top-k logits to shards.
           One-shot, ~22 h on T4 for 500 MB corpus. ~14 GB on disk.

  train  - train a student against the cached logits using KL + CE blend:
              L = alpha * CE(student, true_next) +
                  (1 - alpha) * T^2 * KL(student || teacher_topk)

Vocabulary note: cross-vocab distillation is hard. For this first cut the
student shares the teacher's tokenizer (Phi-3 BPE, vocab=32064). Embedding
table dominates param count - at d_model=256 the embed is ~8.2 M params,
giving a total student around ~12-16 M params. Acceptable for our budget.

Usage:
  # 1) cache teacher logits
  python -m src.train.distill cache \\
      --corpus data/raw/fineweb_edu.txt \\
      --out data/distill/logits \\
      --teacher microsoft/Phi-3-mini-4k-instruct \\
      --topk 20 --max-len 512

  # 2) distill train (mamba hybrid student)
  python -m src.train.distill train \\
      --logit-dir data/distill/logits \\
      --tokenizer-id microsoft/Phi-3-mini-4k-instruct \\
      --attention mamba --d-model 256 --n-layers 6 \\
      --epochs 3 --batch-size 16 \\
      --alpha 0.3 --temperature 2.0 \\
      --ckpt checkpoints/distill_mamba.pt
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model.decoder import Decoder
from src.model.heads import CausalLMHead


# ============================================================================
# CACHE MODE
# ============================================================================

@torch.no_grad()
def cache_teacher_logits(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"teacher: {args.teacher}  device={device}  dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()

    raw = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
    print(f"corpus chars: {len(raw):,}")
    print("tokenizing (may take a minute for large corpora)...")
    ids = tok.encode(raw, add_special_tokens=False)
    print(f"teacher tokens: {len(ids):,}")

    L = args.max_len
    n_blocks = len(ids) // L
    print(f"blocks of len {L}: {n_blocks:,}")
    if n_blocks == 0:
        raise SystemExit("corpus too short for chosen --max-len")

    buf_ids, buf_topk_idx, buf_topk_val = [], [], []
    shard_idx = 0
    t0 = time.time()

    for b in range(n_blocks):
        block = ids[b * L:(b + 1) * L]
        x = torch.tensor([block], dtype=torch.long, device=device)
        out = teacher(x)
        logits = out.logits[0]                       # (L, V)
        logp = F.log_softmax(logits.float(), dim=-1)
        vals, idx = logp.topk(args.topk, dim=-1)     # (L, k), (L, k)
        buf_ids.append(torch.tensor(block, dtype=torch.long))
        buf_topk_idx.append(idx.cpu().to(torch.int32))
        buf_topk_val.append(vals.cpu().to(torch.float16))

        if (b + 1) % args.shard_blocks == 0 or b == n_blocks - 1:
            shard_path = out_dir / f"shard_{shard_idx:05d}.pt"
            torch.save({
                "input_ids": torch.stack(buf_ids),
                "topk_ids":  torch.stack(buf_topk_idx),
                "topk_logp": torch.stack(buf_topk_val),
            }, shard_path)
            dt = time.time() - t0
            tps = (b + 1) * L / max(dt, 1e-3)
            print(f"shard {shard_idx} -> {shard_path.name}  "
                  f"blocks={len(buf_ids)}  tok/s={tps:.0f}  elapsed={dt:.1f}s")
            shard_idx += 1
            buf_ids, buf_topk_idx, buf_topk_val = [], [], []

    meta = {
        "teacher": args.teacher,
        "max_len": L,
        "topk": args.topk,
        "n_blocks": n_blocks,
        "n_shards": shard_idx,
        "vocab_size": teacher.config.vocab_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cached {n_blocks} blocks to {out_dir}")


# ============================================================================
# TRAIN MODE
# ============================================================================

class DistillDataset(Dataset):
    """Lazy-loaded shards of (input_ids, top-k teacher logits)."""

    def __init__(self, logit_dir, shard_cache_size=2):
        self.dir = Path(logit_dir)
        self.shard_paths = sorted(self.dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise RuntimeError(f"no shards in {logit_dir}")

        meta = json.loads((self.dir / "meta.json").read_text())
        self.max_len = meta["max_len"]
        self.topk = meta["topk"]
        self.vocab_size = meta["vocab_size"]
        self.teacher = meta.get("teacher")

        # build (shard_path, local_index) lookup table
        self._index = []
        for p in self.shard_paths:
            # peek without keeping in memory
            head = torch.load(p, map_location="cpu")
            n = head["input_ids"].size(0)
            for i in range(n):
                self._index.append((p, i))
            del head

        self._cache = {}
        self._cache_size = shard_cache_size
        print(f"DistillDataset: {len(self._index):,} blocks, "
              f"{len(self.shard_paths)} shards, V={self.vocab_size}, "
              f"L={self.max_len}, topk={self.topk}")

    def _shard(self, p):
        if p in self._cache:
            return self._cache[p]
        if len(self._cache) >= self._cache_size:
            # evict oldest
            self._cache.pop(next(iter(self._cache)))
        self._cache[p] = torch.load(p, map_location="cpu")
        return self._cache[p]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        p, i = self._index[idx]
        s = self._shard(p)
        return {
            "input_ids": s["input_ids"][i].long(),
            "topk_ids":  s["topk_ids"][i].long(),
            "topk_logp": s["topk_logp"][i].float(),
        }


def distill_loss(student_logits, labels, topk_ids, topk_logp, *,
                 alpha=0.3, temperature=2.0):
    """L = alpha * CE(student, label) + (1-alpha) * T^2 * KL(student||teacher_topk).

    student_logits: (B, L, V)
    labels:         (B, L)         -100 to ignore
    topk_ids:       (B, L, k)      teacher's top-k token ids
    topk_logp:      (B, L, k)      teacher log-probs at those ids (log-softmax)
    """
    V = student_logits.size(-1)

    ce = F.cross_entropy(
        student_logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100,
    )

    # student log-probs at temperature T, gathered at teacher's top-k positions
    student_logp_T = F.log_softmax(student_logits / temperature, dim=-1)
    student_topk_logp = student_logp_T.gather(-1, topk_ids)        # (B, L, k)

    # teacher probabilities renormalized across its top-k (mass-on-top-k assumption)
    teacher_p = F.softmax(topk_logp, dim=-1)                       # (B, L, k)
    kl = (teacher_p * (topk_logp - student_topk_logp)).sum(dim=-1).mean()
    kl = kl * (temperature ** 2)

    return alpha * ce + (1.0 - alpha) * kl, ce.item(), kl.item()


def train_distill(args):
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    ds = DistillDataset(args.logit_dir)
    if ds.vocab_size != tok.vocab_size:
        raise SystemExit(
            f"vocab mismatch: cache_vocab={ds.vocab_size} "
            f"tokenizer_vocab={tok.vocab_size}. Use the same teacher tokenizer "
            f"for caching and training."
        )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True,
    )

    decoder = Decoder(
        vocab_size=ds.vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_len=ds.max_len,
        attention=args.attention, dropout=args.dropout, pad_id=pad_id,
    ).to(device)
    clm = CausalLMHead(args.d_model, ds.vocab_size,
                       tied_weight=decoder.embed.tok.weight).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"student params: {n_dec:,}  attention={args.attention}  V={ds.vocab_size}")

    # dedupe tied params before optimizer construction
    seen, params = set(), []
    for p in list(decoder.parameters()) + list(clm.parameters()):
        if id(p) in seen:
            continue
        seen.add(id(p)); params.append(p)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))

    total_steps = args.epochs * len(loader)
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * \
               0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    decoder.train(); clm.train()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)     # (B, L)
            topk_ids  = batch["topk_ids"].to(device)      # (B, L, k)
            topk_logp = batch["topk_logp"].to(device)     # (B, L, k)

            # next-token shift: predict t+1 from t. Drop final position from
            # inputs+teacher, drop first from labels.
            inp = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous().clone()
            labels[labels == pad_id] = -100
            # teacher logits at position t describe the distribution OVER token t+1
            # only if the teacher was conditioned on tokens [<bos>..t-1]. In our
            # cache we ran teacher(block) so logits[t] = p(token at t+1 | block[:t+1]).
            # That's exactly what we want for predicting labels[t] = block[t+1].
            tk_ids = topk_ids[:, :-1].contiguous()
            tk_lp = topk_logp[:, :-1].contiguous()

            hidden = decoder(inp)
            logits = clm(hidden)
            loss, ce_val, kl_val = distill_loss(
                logits, labels, tk_ids, tk_lp,
                alpha=args.alpha, temperature=args.temperature,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                tps = (step + 1) * args.batch_size * ds.max_len / max(dt, 1e-3)
                ppl = math.exp(min(ce_val, 20))
                print(f"[ep {ep} step {step:5d}] loss={loss.item():.4f}  "
                      f"ce={ce_val:.4f} (ppl={ppl:.1f})  kl={kl_val:.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  tok/s={tps:.0f}  "
                      f"elapsed={dt:.0f}s")
            step += 1

    out = Path(args.ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "decoder": decoder.state_dict(),
        "clm_head": clm.state_dict(),
        "config": {
            "vocab_size": ds.vocab_size, "d_model": args.d_model,
            "n_heads": args.n_heads, "n_layers": args.n_layers,
            "d_ff": args.d_ff, "max_len": ds.max_len,
            "attention": args.attention, "pad_id": pad_id,
        },
        "tokenizer_id": args.tokenizer_id,
        "meta": {
            "distilled": True,
            "teacher": ds.teacher,
            "teacher_topk": ds.topk,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "epochs": args.epochs,
        },
    }, out)
    print(f"saved checkpoint -> {out}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # ---- cache ----
    cp = sub.add_parser("cache", help="run teacher, save top-k logits to shards")
    cp.add_argument("--corpus", required=True)
    cp.add_argument("--out", default="data/distill/logits")
    cp.add_argument("--teacher", default="microsoft/Phi-3-mini-4k-instruct")
    cp.add_argument("--max-len", type=int, default=512)
    cp.add_argument("--topk", type=int, default=20)
    cp.add_argument("--shard-blocks", type=int, default=1024,
                    help="number of (L,)-length blocks per shard file")

    # ---- train ----
    tp = sub.add_parser("train", help="distill student from cached logits")
    tp.add_argument("--logit-dir", required=True)
    tp.add_argument("--tokenizer-id", default="microsoft/Phi-3-mini-4k-instruct")
    tp.add_argument("--attention", default="mamba",
                    choices=["softmax", "linear", "rwkv", "mamba"])
    tp.add_argument("--d-model", type=int, default=256)
    tp.add_argument("--n-heads", type=int, default=4)
    tp.add_argument("--n-layers", type=int, default=6)
    tp.add_argument("--d-ff", type=int, default=1024)
    tp.add_argument("--dropout", type=float, default=0.1)
    tp.add_argument("--epochs", type=int, default=3)
    tp.add_argument("--batch-size", type=int, default=16)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--warmup-steps", type=int, default=200)
    tp.add_argument("--min-lr-ratio", type=float, default=0.1)
    tp.add_argument("--alpha", type=float, default=0.3,
                    help="weight on hard CE; (1-alpha) on soft KL")
    tp.add_argument("--temperature", type=float, default=2.0,
                    help="softmax temperature for KL distillation")
    tp.add_argument("--log-every", type=int, default=20)
    tp.add_argument("--num-workers", type=int, default=2)
    tp.add_argument("--ckpt", default="checkpoints/distill.pt")

    args = ap.parse_args()
    if args.mode == "cache":
        cache_teacher_logits(args)
    else:
        train_distill(args)


if __name__ == "__main__":
    main()
