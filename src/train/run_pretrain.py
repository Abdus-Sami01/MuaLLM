"""Run MLM pretrain on a real text corpus + saved tokenizer.

Usage:
  python -m src.train.run_pretrain \
    --corpus data/raw/wiki_edu_small.txt \
    --tokenizer data/processed/tokenizer.json \
    --attention softmax --max-steps 200 --batch-size 4
"""
import argparse
import time
from pathlib import Path

import torch

from src.tokenizer.train_bpe import load_tokenizer, SPECIAL_TOKENS
from src.data.chunk import chunk_text
from src.data.pack_tokens import load_meta
from src.model.decoder import Decoder
from src.model.heads import CausalLMHead
from src.train.train_clm import CLMDataset, PackedCLMDataset, save_checkpoint
from torch.utils.data import DataLoader
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",
                    help="raw text corpus (in-RAM path; omit if --packed-bin)")
    ap.add_argument("--packed-bin",
                    help="memmapped token .bin from src.data.pack_tokens "
                         "(streams from disk; preferred for large corpora)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--attention", default="softmax",
                    choices=["softmax", "linear", "rwkv"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--chunk-tokens", type=int, default=126)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    pad_id = tok.token_to_id("[PAD]")
    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    mask_id = tok.token_to_id("[MASK]")
    special_ids = [tok.token_to_id(t) for t in SPECIAL_TOKENS]
    print(f"vocab={vocab_size} pad={pad_id} cls={cls_id} sep={sep_id} mask={mask_id}")

    if not args.packed_bin and not args.corpus:
        raise SystemExit("pass --packed-bin (preferred) or --corpus")

    if args.packed_bin:
        # streaming path: memmap a flat token .bin, slice windows on demand.
        meta = load_meta(args.packed_bin)
        dtype = meta.get("dtype", "uint16")
        n_tokens = meta.get("n_tokens")
        body_len = args.max_len - 2
        step = max(1, body_len - args.stride)  # --stride keeps overlap meaning
        ds = PackedCLMDataset(
            args.packed_bin, cls_id=cls_id, sep_id=sep_id, pad_id=pad_id,
            max_len=args.max_len, step=step, dtype=dtype, n_tokens=n_tokens,
        )
        print(f"packed: {args.packed_bin}  tokens={ds.n_tokens:,}  "
              f"windows={len(ds):,}  step={step}  dtype={dtype}")
        if len(ds) == 0:
            raise SystemExit("no windows; corpus too small or max_len too large")
    else:
        # in-RAM path: load whole corpus, chunk fully into memory.
        print(f"loading corpus: {args.corpus}")
        text = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
        print(f"corpus chars: {len(text):,}")

        print(f"chunking (max={args.chunk_tokens}, stride={args.stride})...")
        chunks = chunk_text(text, tok, max_tokens=args.chunk_tokens,
                            stride=args.stride)
        print(f"chunks: {len(chunks):,}")
        if not chunks:
            raise SystemExit("no chunks produced")

        ds = CLMDataset(chunks, vocab_size, pad_id, cls_id, sep_id,
                        special_ids=special_ids, max_len=args.max_len)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    dec = Decoder(
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_len=args.max_len,
        attention=args.attention, dropout=0.1, pad_id=pad_id,
    )
    clm = CausalLMHead(args.d_model, vocab_size, tied_weight=dec.embed.tok.weight)
    n_dec = sum(p.numel() for p in dec.parameters())
    n_clm = sum(p.numel() for p in clm.parameters() if id(p) != id(dec.embed.tok.weight))
    print(f"params: decoder={n_dec:,}  clm_extra={n_clm:,}  attention={args.attention}")

    # dedupe tied params
    seen, params = set(), []
    for p in list(dec.parameters()) + list(clm.parameters()):
        if id(p) in seen:
            continue
        seen.add(id(p))
        params.append(p)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    dec.train()
    clm.train()
    losses = []
    t0 = time.time()
    step = 0
    done = False
    while not done:
        for batch in loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            hidden = dec(input_ids)
            logits = clm(hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
            if step % args.log_every == 0:
                dt = time.time() - t0
                ips = (step + 1) * args.batch_size * args.max_len / max(dt, 1e-3)
                print(f"[step {step:5d}] loss={loss.item():.4f}  "
                      f"tok/s={ips:.0f}  elapsed={dt:.1f}s")
            step += 1
            if step >= args.max_steps:
                done = True
                break

    dt = time.time() - t0
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    print(f"\nfinished: {step} steps in {dt:.1f}s")
    print(f"loss: {early:.3f} -> {late:.3f}")

    if args.ckpt:
        save_checkpoint(args.ckpt, dec, clm, opt,
                        meta={"attention": args.attention, "steps": step,
                              "loss_early": early, "loss_late": late})
        print(f"saved checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
