"""Local Phi-3-mini synthetic QA generator. Replaces Azure OpenAI version.

Generates teaching-domain Q&A pairs as JSONL on local GPU/CPU. Free aside from
the GPU/CPU hours - no API spend. Loops until target count reached, dedupes by
question hash.

Usage:
  # small smoke
  python generate_qa.py --target 50 --device cpu

  # full run on GPU
  python generate_qa.py --target 5000 --per-call 50 \
      --out data/qa/pk_teaching_qa.jsonl

  # custom model
  python generate_qa.py --model microsoft/Phi-3-mini-4k-instruct --target 1000
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


SYSTEM = (
    "You are a strict data generation assistant. "
    "Output only raw JSONL with no commentary, no markdown fences, no preamble."
)

USER_TEMPLATE = """Generate {n} highly realistic, high-quality Q&A pairs for an AI Chatbot designed to help teachers in Pakistan.

Vary topics across these areas:
- Classroom management for large class sizes (50+ students)
- Single National Curriculum (SNC) and provincial alternatives
- FBISE, BISE, matric, intermediate exam patterns
- Translanguaging (Urdu/Punjabi to English medium transition)
- PTM (Parent-Teacher Meeting) interactions in Pakistani context
- Corporal punishment alternatives and positive discipline
- Load shedding and lack of resources in public schools
- Lesson planning, formative vs summative assessment
- Inclusive education for students with diverse learning needs
- Professional development and teacher certification

Output format: every line is one valid JSON object of this exact shape:
{{"text": "User: <question>\\nBot: <answer> [SEP]"}}

Do not output markdown code blocks. Do not number the lines. Do not add commentary."""


def load_teacher(model_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, n_pairs, device, max_new_tokens=4096,
                   temperature=0.9, top_p=0.95):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(n=n_pairs)},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].size(1):]
    return tok.decode(new_tokens, skip_special_tokens=True)


CODE_FENCE_RE = re.compile(r"```(?:json|jsonl)?\s*", re.IGNORECASE)


def parse_jsonl(text):
    # strip code fences if model added them
    text = CODE_FENCE_RE.sub("", text).replace("```", "")
    pairs = []
    for raw in text.splitlines():
        line = raw.strip().rstrip(",")
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("text", "")
        if "User:" in t and "Bot:" in t:
            pairs.append({"text": t})
    return pairs


def question_hash(text):
    q = text.split("Bot:")[0].strip().lower()
    return hashlib.md5(q.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000,
                    help="target number of unique pairs in output file")
    ap.add_argument("--per-call", type=int, default=50,
                    help="how many pairs requested per Phi-3 generation call")
    ap.add_argument("--out", default="data/qa/pk_teaching_qa.jsonl")
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu. default: cuda if available")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-calls", type=int, default=500,
                    help="hard cap on generation calls (safety)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}  model={args.model}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dedupe against existing pairs in output file
    seen = set()
    if out_path.exists():
        for raw in out_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                seen.add(question_hash(obj["text"]))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"existing unique pairs: {len(seen)}")
    if len(seen) >= args.target:
        print("target already met. nothing to do.")
        return

    print("loading teacher model...")
    tok, model = load_teacher(args.model, device, dtype)

    added = 0
    calls = 0
    stale_calls = 0
    temp = args.temperature
    with open(out_path, "a", encoding="utf-8") as f:
        while len(seen) < args.target and calls < args.max_calls:
            calls += 1
            text = generate_batch(
                tok, model, args.per_call, device,
                max_new_tokens=args.max_new_tokens,
                temperature=temp, top_p=args.top_p,
            )
            pairs = parse_jsonl(text)
            new_this_call = 0
            for p in pairs:
                h = question_hash(p["text"])
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                added += 1
                new_this_call += 1
                if len(seen) >= args.target:
                    break
            f.flush()
            print(f"[call {calls}] new={new_this_call}  "
                  f"total={len(seen)}/{args.target}  temp={temp:.2f}")

            if new_this_call == 0:
                stale_calls += 1
                if stale_calls >= 3 and temp < 1.3:
                    temp = round(temp + 0.1, 2)
                    print(f"  raising temperature -> {temp}")
                    stale_calls = 0
            else:
                stale_calls = 0

    print(f"done. added {added} pairs in {calls} calls. file: {out_path}")


if __name__ == "__main__":
    main()
