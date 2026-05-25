"""Stream FineWeb-Edu and write a filtered text corpus.

FineWeb-Edu (HuggingFaceFW/fineweb-edu) is ~1.3T tokens of education-filtered
Common Crawl, scored 0-5 by an educational-quality classifier. We stream the
sample subset and write docs above a score threshold to a single text file,
optionally narrowing further by keyword (e.g. teacher/classroom).

This replaces the old src/data/download_wiki.py for compute-optimal pretraining
(your previous 50 MB wiki was ~12 M tokens; 8 M-param model needs ~160 M for
Chinchilla. FineWeb-Edu lets us scale to 500 MB cleanly).

Usage:
  # 500 MB general edu corpus
  python -m src.data.fineweb_edu --out data/raw/fineweb_edu.txt --max-mb 500

  # tighter teaching-focused slice
  python -m src.data.fineweb_edu \\
      --out data/raw/fineweb_edu_teach.txt --max-mb 500 \\
      --keywords teacher,classroom,curriculum,pedagogy,lesson

  # bigger subset for full distill
  python -m src.data.fineweb_edu --subset sample-100BT --max-mb 2000
"""
import argparse
from pathlib import Path


VALID_SUBSETS = [
    "sample-10BT", "sample-100BT", "sample-350BT",
    "CC-MAIN-2024-10", "CC-MAIN-2024-18", "CC-MAIN-2024-22",
]


def stream_fineweb_edu(out_path, max_bytes, *, subset="sample-10BT",
                       keywords=None, min_score=3.0, log_every=200):
    """Stream FineWeb-Edu docs, write text to out_path. Returns stats dict.

    Each line in the output file is followed by a blank-line separator.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=subset,
        split="train", streaming=True,
    )

    kw = [k.lower() for k in (keywords or [])]
    written = 0
    n_docs = 0
    n_skip_score = 0
    n_skip_kw = 0

    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or ""
            if not text:
                continue
            score = float(row.get("score") or 0.0)
            if score < min_score:
                n_skip_score += 1
                continue
            if kw:
                low = text.lower()
                if not any(k in low for k in kw):
                    n_skip_kw += 1
                    continue
            f.write(text.strip() + "\n\n")
            written += len(text.encode("utf-8"))
            n_docs += 1
            if n_docs % log_every == 0:
                print(f"  docs={n_docs:,}  MB={written/1e6:.1f}  "
                      f"skip_score={n_skip_score:,}  skip_kw={n_skip_kw:,}")
            if written >= max_bytes:
                break

    return {
        "bytes": written,
        "mb": round(written / 1e6, 2),
        "docs": n_docs,
        "skip_score": n_skip_score,
        "skip_kw": n_skip_kw,
        "min_score": min_score,
        "subset": subset,
        "path": str(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/fineweb_edu.txt")
    ap.add_argument("--max-mb", type=int, default=500)
    ap.add_argument("--subset", default="sample-10BT", choices=VALID_SUBSETS)
    ap.add_argument("--keywords", default="",
                    help="comma-separated keyword filter (optional, AND with score)")
    ap.add_argument("--min-score", type=float, default=3.0,
                    help="FineWeb-Edu quality score threshold (0-5). 3 is recommended.")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or None
    stats = stream_fineweb_edu(
        args.out, max_bytes=args.max_mb * 1024 * 1024,
        subset=args.subset, keywords=keywords,
        min_score=args.min_score, log_every=args.log_every,
    )
    print("\nfinal:")
    for k, v in stats.items():
        print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
