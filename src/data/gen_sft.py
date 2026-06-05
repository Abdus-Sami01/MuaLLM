"""Generate SFT / distillation data from a LOCAL open-weight instruct teacher.

No API. The teacher is a Hugging Face model run on your own machine (Colab /
T4 / CPU). It writes two interchangeable artifacts from the same run:

  --out  data/qa/edu_seqkd.jsonl   {"text": "User: <q>\\nBot: <a> [SEP]"}
                                   -> consumed directly by src.train.run_sft
  --txt  data/raw/edu_seqkd.txt    same pairs as plain text, blank-line sep
                                   -> corpus for src.train.distill cache
                                      (logit-KD) or src.train.run_pretrain

Why this is the cheap path: a tiny from-scratch model cannot learn language
from a few thousand hard labels. A local instruct teacher hands you coherent,
in-domain (instruction -> answer) text; the student then either plain-SFTs on
it (cheapest) or logit-distills against it (more sample-efficient). The output
embeds chat.py's "User:/Bot:" template and a trailing [SEP] so the student
learns where to stop.

Teacher pick (vocab sets the distill student's embed size; see distill.py):
  HuggingFaceTB/SmolLM2-360M-Instruct  49k vocab, 360M -> fast cache (default)
  microsoft/Phi-3-mini-4k-instruct     32k vocab, 3.8B -> smallest student, slow
  Qwen/Qwen2.5-0.5B-Instruct          152k vocab -> big embed, avoid for distill

Usage:
  python -m src.data.gen_sft \\
      --teacher HuggingFaceTB/SmolLM2-360M-Instruct \\
      --n 3000 --out data/qa/edu_seqkd.jsonl --txt data/raw/edu_seqkd.txt
"""
import argparse
import json
import random
from pathlib import Path

import torch


# Education / teaching domain. Compose <template> x <topic> for varied prompts.
TEMPLATES = [
    "Explain {t} to a new teacher in simple terms.",
    "What is {t} and why does it matter in the classroom?",
    "Give two practical classroom strategies for {t}.",
    "A student asks about {t}. How would you answer?",
    "Summarize the key idea behind {t} in a few sentences.",
    "What are common mistakes teachers make with {t}?",
    "How does {t} affect student learning?",
    "Describe a short example of {t} in a lesson.",
]

TOPICS = [
    "formative assessment", "differentiated instruction", "classroom management",
    "scaffolding", "Bloom's taxonomy", "active learning", "growth mindset",
    "spaced repetition", "the zone of proximal development", "inclusive education",
    "project-based learning", "questioning techniques", "lesson planning",
    "student feedback", "phonics instruction", "reading comprehension",
    "metacognition", "peer learning", "rubrics", "learning objectives",
    "behavior reinforcement", "special educational needs", "flipped classroom",
    "concept mapping", "retrieval practice", "the forgetting curve",
    "intrinsic motivation", "classroom discussion", "summative assessment",
    "multimodal teaching", "early literacy", "numeracy foundations",
    "emotional regulation in children", "teacher-student rapport",
    "homework design", "group work dynamics", "direct instruction",
    "inquiry-based learning", "assessment for learning", "curriculum sequencing",
]


def build_seed_prompts(n, seeds_file=None, rng=None):
    """Return n prompt strings. From a seeds file (one per line) or composed."""
    rng = rng or random.Random(0)
    if seeds_file:
        lines = [l.strip() for l in Path(seeds_file).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise SystemExit(f"no seeds in {seeds_file}")
        return [rng.choice(lines) for _ in range(n)]
    combos = [tpl.format(t=t) for tpl in TEMPLATES for t in TOPICS]
    rng.shuffle(combos)
    if n <= len(combos):
        return combos[:n]
    # more samples than unique combos: repeat; teacher sampling varies answers
    return [combos[i % len(combos)] for i in range(n)]


@torch.no_grad()
def generate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 (8-bit exponent like fp32) is sampling-safe AND half the memory of
    # fp32, so it fits bigger teachers/batches on any CUDA GPU incl. T4. fp16 is
    # NOT safe here: 5-bit exponent overflows -> inf logits -> multinomial assert.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"teacher: {args.teacher}  device={device}  dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad so generated tokens align at the right
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    rng = random.Random(args.seed)
    prompts = build_seed_prompts(args.n, args.seeds_file, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = Path(args.txt) if args.txt else None
    if txt_path:
        txt_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_written = 0
    out_f = out_path.open("w", encoding="utf-8")
    txt_f = txt_path.open("w", encoding="utf-8") if txt_path else None
    try:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            chats = [
                tok.apply_chat_template(
                    ([{"role": "system", "content": args.system}] if args.system
                     else []) + [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tok(chats, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len).to(device)
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                renormalize_logits=True, pad_token_id=tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]  # strip the prompt
            answers = tok.batch_decode(new, skip_special_tokens=True)

            for q, a in zip(batch, answers):
                a = a.strip()
                if not a:
                    continue
                key = (q, a)
                if key in seen:
                    continue
                seen.add(key)
                # JSONL is for BPE-tokenizer SFT (run_sft): literal [SEP] maps to
                # the project stop token. TXT is for HF-tokenizer distillation
                # (distill cache injects the teacher EOS between these blank-line
                # docs), so it carries no [SEP] marker.
                out_f.write(json.dumps(
                    {"text": f"User: {q}\nBot: {a} [SEP]"},
                    ensure_ascii=False) + "\n")
                if txt_f:
                    txt_f.write(f"User: {q}\nBot: {a}\n\n")
                n_written += 1
            out_f.flush()
            print(f"  [{min(start + args.batch_size, len(prompts)):5d}/"
                  f"{len(prompts)}] written={n_written}")
    finally:
        out_f.close()
        if txt_f:
            txt_f.close()

    print(f"\nwrote {n_written} pairs -> {out_path}")
    if txt_path:
        print(f"plain corpus -> {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--n", type=int, default=3000, help="number of pairs to gen")
    ap.add_argument("--out", default="data/qa/edu_seqkd.jsonl",
                    help="JSONL for src.train.run_sft")
    ap.add_argument("--txt", default=None,
                    help="optional plain-text corpus for src.train.distill")
    ap.add_argument("--seeds-file", default=None,
                    help="optional file of seed prompts (one per line); "
                         "overrides the built-in education templates")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512, help="prompt truncation")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--system",
                    default="You are a knowledgeable teaching assistant. Answer "
                            "the question directly and concisely. Do not "
                            "introduce yourself or mention being an AI.",
                    help="system prompt for the teacher (pass '' to disable)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
