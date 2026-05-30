"""Pack a text corpus into a flat memmapped token stream.

Replaces the in-RAM `read_text` + `chunk_text` path for large corpora. The old
path held BOTH the full corpus string AND every overlapping token chunk
(`list[list[int]]`, ~28 B per Python int) in RAM at once -- multiple GB for a
160 M-token Chinchilla run. This streams the corpus through the tokenizer once
and writes the ids to a flat `uint16` `.bin` (2 B/token). Training then memmaps
that file and slices windows on demand (see `PackedCLMDataset` in
`src.train.train_clm`), so peak RAM no longer scales with corpus size.

Why memmap `.bin` and not HDF5: no new dependency (numpy is already in the
stack, h5py is not), and plain memmap is faster for the many tiny random-window
reads a CLM DataLoader issues (no per-read chunk decompression). The trade-off
is no built-in compression -- fine, since uint16 tokens are already compact.

dtype: `uint16` holds ids up to 65535; our BPE vocab is 8k, so it fits with
room to spare. If you raise the vocab past 65535, pass `--dtype uint32`.

Usage:
  python -m src.data.pack_tokens \
    --corpus data/raw/fineweb_edu.txt \
    --tokenizer data/processed/tokenizer.json \
    --out data/processed/fineweb_edu.bin

  # insert a doc separator between documents (defaults to [SEP])
  python -m src.data.pack_tokens ... --doc-sep
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.tokenizer.train_bpe import load_tokenizer


DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def pack_corpus(corpus_path, tokenizer, out_path, *, dtype="uint16",
                doc_sep_id=None, batch_lines=1000, log_every_docs=200_000):
    """Stream `corpus_path` through `tokenizer`, write ids to `out_path` (.bin).

    Lines are stripped; blank lines skipped. Each non-blank line is encoded
    without special tokens and appended to a flat token stream. If
    `doc_sep_id` is given, it is written after each line as a soft boundary.

    Returns a meta dict; also writes it to `<out_path>.meta.json`.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}, got {dtype!r}")
    np_dtype = DTYPES[dtype]
    maxval = int(np.iinfo(np_dtype).max)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_tokens = 0
    n_docs = 0
    t0 = time.time()

    def encode_and_write(fout, texts):
        nonlocal n_tokens
        if not texts:
            return
        for enc in tokenizer.encode_batch(texts, add_special_tokens=False):
            ids = enc.ids
            if not ids:
                continue
            if doc_sep_id is not None:
                ids = ids + [doc_sep_id]
            arr = np.asarray(ids, dtype=np.int64)
            if int(arr.max()) > maxval:
                raise ValueError(
                    f"token id {int(arr.max())} exceeds {dtype} max {maxval}; "
                    f"use --dtype uint32"
                )
            fout.write(arr.astype(np_dtype).tobytes())
            n_tokens += arr.size

    buf = []
    with open(out, "wb") as fout, \
            open(corpus_path, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            n_docs += 1
            if len(buf) >= batch_lines:
                encode_and_write(fout, buf)
                buf = []
            if n_docs % log_every_docs == 0:
                dt = time.time() - t0
                print(f"  docs={n_docs:,}  tokens={n_tokens:,}  "
                      f"MB={n_tokens * np_dtype().itemsize / 1e6:.1f}  "
                      f"elapsed={dt:.1f}s")
        encode_and_write(fout, buf)

    meta = {
        "path": str(out),
        "n_tokens": int(n_tokens),
        "n_docs": int(n_docs),
        "dtype": dtype,
        "itemsize": int(np_dtype().itemsize),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "doc_sep_id": doc_sep_id,
        "bytes": int(n_tokens * np_dtype().itemsize),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_meta(bin_path):
    """Load the sidecar meta for a packed .bin, or {} if absent."""
    p = Path(str(bin_path) + ".meta.json")
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="uint16", choices=list(DTYPES))
    ap.add_argument("--doc-sep", action="store_true",
                    help="write [SEP] id after each doc as a soft boundary")
    ap.add_argument("--batch-lines", type=int, default=1000)
    args = ap.parse_args()

    print(f"loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)
    doc_sep_id = tok.token_to_id("[SEP]") if args.doc_sep else None

    print(f"packing corpus: {args.corpus} -> {args.out}  (dtype={args.dtype})")
    meta = pack_corpus(args.corpus, tok, args.out, dtype=args.dtype,
                       doc_sep_id=doc_sep_id, batch_lines=args.batch_lines)
    print("\nfinal:")
    for k, v in meta.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
