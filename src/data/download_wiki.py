"""Pull a small slice of Wikipedia (education-related articles) for pretraining.

Uses `datasets` to stream wikipedia and filter by keyword. No API key.
For semester project: cap at ~50 MB so CPU pretrain finishes in days, not weeks.
"""
from pathlib import Path

KEYWORDS = [
    "teacher", "teaching", "education", "pedagogy", "classroom",
    "curriculum", "lesson plan", "school", "learning", "assessment",
    "instruction", "literacy", "numeracy", "kindergarten",
]


def download_subset(out_path, max_bytes=50 * 1024 * 1024, lang="en", max_articles=5000):
    """Stream wikipedia, write filtered articles to a single text file."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True,
    )

    written = 0
    n_articles = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            title = (row.get("title") or "").lower()
            text = row.get("text") or ""
            if not text:
                continue
            if not any(k in title for k in KEYWORDS):
                continue
            f.write(text.strip() + "\n\n")
            written += len(text.encode("utf-8"))
            n_articles += 1
            if written >= max_bytes or n_articles >= max_articles:
                break
    return {"bytes": written, "articles": n_articles, "path": str(out)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wiki_edu.txt")
    ap.add_argument("--max-mb", type=int, default=50)
    ap.add_argument("--max-articles", type=int, default=5000)
    args = ap.parse_args()
    stats = download_subset(args.out, max_bytes=args.max_mb * 1024 * 1024,
                            max_articles=args.max_articles)
    print(stats)
