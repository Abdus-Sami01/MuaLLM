"""Rule-based synthetic QA pair generator from plain text.

Strategy: for each sentence with a named-entity-ish surface form (number, date,
proper noun, capitalised phrase), generate a templated question and use the
surface form as the answer span.

This is the cheap-and-honest augmentation. Hand-written QA pairs go in
data/qa/manual.json separately.
"""
import json
import re
from pathlib import Path


NUMBER_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def gen_from_sentence(sent):
    pairs = []
    for m in YEAR_RE.finditer(sent):
        ans = m.group(0)
        q = f"In what year {strip_subject(sent, ans)}?"
        pairs.append({"question": q, "context": sent, "answer": ans,
                      "answer_start": m.start()})
    for m in NUMBER_RE.finditer(sent):
        ans = m.group(0)
        if YEAR_RE.fullmatch(ans):
            continue
        q = f"How many {strip_subject(sent, ans)}?"
        pairs.append({"question": q, "context": sent, "answer": ans,
                      "answer_start": m.start()})
    for m in PROPER_NOUN_RE.finditer(sent):
        ans = m.group(0)
        if len(ans.split()) > 4:
            continue
        q = f"Who or what {strip_subject(sent, ans)}?"
        pairs.append({"question": q, "context": sent, "answer": ans,
                      "answer_start": m.start()})
    return pairs


def strip_subject(sent, ans):
    s = sent.replace(ans, "____").rstrip(".!?")
    return s[:120]


def generate(text, max_per_sentence=2):
    out = []
    for sent in split_sentences(text):
        pairs = gen_from_sentence(sent)[:max_per_sentence]
        out.extend(pairs)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-sentence", type=int, default=2)
    args = ap.parse_args()
    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    pairs = generate(text, max_per_sentence=args.max_per_sentence)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"generated {len(pairs)} pairs -> {args.out}")
