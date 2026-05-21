"""End-to-end smoke test. Runs all 3 attention variants on a toy corpus.

Validates: tokenizer trains -> model builds -> MLM forward/backward -> loss drops.
Should finish in ~1-2 minutes on CPU.
"""
import sys
import tempfile
from pathlib import Path

import torch

from src.tokenizer.train_bpe import train_bpe, SPECIAL_TOKENS
from src.data.chunk import chunk_text
from src.model.decoder import Decoder
from src.model.heads import CausalLMHead
from src.train.train_clm import CLMDataset, train_clm


TOY_CORPUS = """
Teachers play a fundamental role in shaping the minds of future generations.
Effective teaching requires patience, subject expertise, and emotional intelligence.
Classroom management is one of the most challenging aspects of the teaching profession.
Lesson planning involves setting clear learning objectives and selecting activities.
Assessment can be formative or summative depending on the purpose.
Differentiated instruction tailors teaching to the diverse needs of learners.
Professional development is essential for ongoing teacher growth and certification.
Building rapport with students improves engagement and learning outcomes.
A reflective practitioner regularly evaluates their own teaching methods.
The teaching profession demands strong communication and interpersonal skills.
Curriculum design balances content coverage with skill development.
Formative assessments inform instruction during a lesson, not just after.
Inclusive classrooms support students with diverse learning needs.
"""


def run_variant(attention, seed=42):
    torch.manual_seed(seed)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus_file = tmp / "corpus.txt"
        corpus_file.write_text(TOY_CORPUS * 50, encoding="utf-8")

        tok_path = tmp / "tokenizer.json"
        tokenizer = train_bpe([corpus_file], tok_path,
                              vocab_size=800, min_frequency=1)
        text = TOY_CORPUS * 50
        chunks = chunk_text(text, tokenizer, max_tokens=60, stride=15)
        if not chunks:
            raise RuntimeError("no chunks produced")

        vocab_size = tokenizer.get_vocab_size()
        pad_id = tokenizer.token_to_id("[PAD]")
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        mask_id = tokenizer.token_to_id("[MASK]")
        special_ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS]

        decoder = Decoder(
            vocab_size=vocab_size, d_model=64, n_heads=2,
            n_layers=2, d_ff=128, max_len=80,
            attention=attention, dropout=0.0, pad_id=pad_id,
        )
        clm = CausalLMHead(d_model=64, vocab_size=vocab_size,
                      tied_weight=decoder.embed.tok.weight)

        n_params = sum(p.numel() for p in decoder.parameters()) + \
                   sum(p.numel() for p in clm.parameters())

        ds = CLMDataset(chunks, vocab_size, pad_id, cls_id, sep_id,
                        special_ids=special_ids, max_len=64)

        losses = train_clm(decoder, clm, ds, epochs=4, batch_size=4,
                           lr=3e-4, log_every=999)

        n = max(3, len(losses) // 5)
        early = sum(losses[:n]) / n
        late = sum(losses[-n:]) / n
        return {
            "attention": attention,
            "params": n_params,
            "chunks": len(chunks),
            "steps": len(losses),
            "loss_early": early,
            "loss_late": late,
            "improved": late < early,
        }


def main():
    variants = sys.argv[1:] or ["softmax", "linear", "rwkv"]
    print(f"running variants: {variants}\n")
    results = []
    for v in variants:
        print(f"=== {v} ===")
        r = run_variant(v)
        print(f"  params: {r['params']:,}  chunks: {r['chunks']}  steps: {r['steps']}")
        print(f"  loss: {r['loss_early']:.3f} -> {r['loss_late']:.3f}  "
              f"({'OK' if r['improved'] else 'FAIL'})\n")
        results.append(r)

    print("=" * 40)
    fail = [r for r in results if not r["improved"]]
    if fail:
        print(f"FAIL: {[r['attention'] for r in fail]} loss did not decrease")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
