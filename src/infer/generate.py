"""Autoregressive generation / chat with a trained causal LM checkpoint.

Handles both checkpoint flavors transparently:
  * SFT / pretrain  -> stores `tokenizer_path` (project BPE, [CLS]/[SEP])
  * distilled        -> stores `tokenizer_id`  (teacher HF tokenizer, BOS/EOS)

`TokAdapter` normalizes the two tokenizer APIs (project `tokenizers.Tokenizer`
vs HF `AutoTokenizer`) behind encode/decode + bos_id/eos_id so the generation
loop is identical for both.

Usage:
  # single question
  python -m src.infer.generate --ckpt checkpoints/edu_distill.pt \
      --prompt "What is formative assessment?"

  # interactive chat loop
  python -m src.infer.generate --ckpt checkpoints/edu_distill.pt
"""
import argparse
import re

import torch
import torch.nn.functional as F

from src.model.decoder import Decoder
from src.model.heads import CausalLMHead


class TokAdapter:
    """Uniform encode/decode + bos/eos over project-BPE or HF tokenizers."""

    def __init__(self, kind, tok, bos_id, eos_id):
        self.kind = kind          # "bpe" | "hf"
        self.tok = tok
        self.bos_id = bos_id
        self.eos_id = eos_id

    def encode(self, text):
        if self.kind == "bpe":
            return self.tok.encode(text, add_special_tokens=False).ids
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        if self.kind == "bpe":
            return self.tok.decode(ids)
        return self.tok.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size() if self.kind == "bpe" else self.tok.vocab_size


def build_tokenizer(ck, override_path=None):
    """Build a TokAdapter from checkpoint fields (or a forced BPE path)."""
    meta = ck.get("meta", {})
    tok_path = override_path or ck.get("tokenizer_path") or meta.get("tokenizer_path")
    tok_id = ck.get("tokenizer_id") or meta.get("tokenizer_id")

    if tok_path:
        from src.tokenizer.train_bpe import load_tokenizer
        t = load_tokenizer(tok_path)
        return TokAdapter("bpe", t, t.token_to_id("[CLS]"), t.token_to_id("[SEP]"))
    if tok_id:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
        bos = t.bos_token_id if t.bos_token_id is not None else t.eos_token_id
        return TokAdapter("hf", t, bos, t.eos_token_id)
    raise SystemExit(
        "checkpoint has no tokenizer_path or tokenizer_id; pass --tokenizer"
    )


@torch.no_grad()
def generate(decoder, clm_head, tok, prompt, *, max_new_tokens=120,
             temperature=0.7, top_k=40, min_new_tokens=8,
             repetition_penalty=1.3, no_repeat_ngram_size=3,
             stop_strings=("\nUser:", "\nBot:", "\nAssistant:"),
             device="cpu"):
    """Top-k sampled continuation with guards that keep a tiny student readable.

    A small distilled student is fluent and mostly on-topic but degenerates at
    decode time: it loops phrases/sentences, leaks training structure (re-emits
    "Bot:"/"User:" turns), or stops instantly with an empty reply. These guards
    fix those *presentation* failures with no retrain; they do NOT create global
    coherence, which comes from training the student enough (epochs) on enough
    in-domain data.

      * min_new_tokens       - block EOS for the first N steps -> no empty reply
      * repetition_penalty   - damp logits of already-emitted tokens
      * no_repeat_ngram_size - forbid repeating any n-gram (kills sentence loops)
      * stop_strings         - halt + trim when the model starts a NEW turn

    Stops at the tokenizer eos id, the first stop string, or max_new_tokens.
    """
    decoder.eval()
    clm_head.eval()
    max_len = decoder.embed.max_len

    # BPE/SFT models train with a mandatory [CLS] prefix; distilled HF-tokenizer
    # students train on raw blocks with no leading BOS, so don't force one.
    ids = ([tok.bos_id] if tok.kind == "bpe" else []) + tok.encode(prompt)
    generated = []
    for t in range(max_new_tokens):
        window = ids[-max_len:]
        inp = torch.tensor([window], dtype=torch.long, device=device)
        logits = clm_head(decoder(inp))[0, -1, :]

        # repetition penalty (HF-style: push seen-token logits toward -inf)
        if repetition_penalty and repetition_penalty != 1.0 and generated:
            seen = torch.tensor(sorted(set(generated)), device=logits.device)
            v = logits.index_select(0, seen)
            v = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)
            logits.index_copy_(0, seen, v)

        # no-repeat n-gram: forbid completing an n-gram already emitted. This is
        # what actually breaks the sentence-level loops a token penalty barely
        # dents (e.g. "...at increasingly longer intervals" repeated verbatim).
        n = no_repeat_ngram_size
        if n and len(generated) >= n:
            prefix = tuple(generated[-(n - 1):])
            for i in range(len(generated) - n + 1):
                if tuple(generated[i:i + n - 1]) == prefix:
                    logits[generated[i + n - 1]] = float("-inf")

        logits = logits / max(temperature, 1e-5)

        # don't allow a stop before min_new_tokens -> kills empty / 1-word replies
        if t < min_new_tokens and tok.eos_id is not None:
            logits[tok.eos_id] = float("-inf")

        if top_k:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == tok.eos_id:
            break
        ids.append(nxt)
        generated.append(nxt)

        # Stop if the student leaks into a NEW turn. Strip a leading echoed
        # marker FIRST, otherwise the prompt's own trailing "Bot:" that the
        # student parrots back would match a stop string at index 0 and truncate
        # the whole answer to empty.
        answer = _strip_markers(tok.decode(generated))
        cut = min((answer.find(s) for s in stop_strings if s in answer), default=-1)
        if cut != -1:
            return _clean(answer[:cut])

    return _clean(tok.decode(generated))


def _strip_markers(text):
    """Trim a leaked leading turn marker (the prompt ends at 'Bot:', so the
    student sometimes echoes it back) plus surrounding whitespace."""
    out = text.strip()
    for m in ("Bot:", "Bot :", "User:", "Assistant:"):
        if out.startswith(m):
            out = out[len(m):].strip()
    return out


def _clean(text):
    """Final cosmetic pass: drop a leading marker echo, collapse blank-line runs."""
    out = _strip_markers(text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def load_model(ckpt_path, device="cpu"):
    """Return (decoder, clm_head, config, raw_ckpt). Tokenizer built separately."""
    ck = torch.load(ckpt_path, map_location=device)
    config = ck.get("config") or ck.get("meta", {}).get("config")
    if config is None:
        raise SystemExit(
            "checkpoint has no 'config'. Re-run training with the fixed scripts "
            "(they store config + tokenizer in the checkpoint)."
        )
    decoder = Decoder(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_len=config["max_len"],
        attention=config["attention"], pad_id=config["pad_id"],
    )
    decoder.load_state_dict(ck["decoder"])
    clm = CausalLMHead(config["d_model"], config["vocab_size"],
                       tied_weight=decoder.embed.tok.weight)
    clm.load_state_dict(ck["clm_head"])
    decoder.to(device)
    clm.to(device)
    return decoder, clm, config, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="override BPE tokenizer path (else read from checkpoint)")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--prompt", default=None,
                    help="single question; if omitted, interactive chat loop")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, clm, config, ck = load_model(args.ckpt, device)
    tok = build_tokenizer(ck, args.tokenizer)
    print(f"loaded {args.ckpt}  attention={config['attention']}  "
          f"tok={tok.kind}  device={device}")

    def answer(question):
        return generate(decoder, clm, tok, f"User: {question}\nBot:",
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        device=device)

    if args.prompt:
        print(f"\nUser: {args.prompt}")
        print(f"Bot: {answer(args.prompt)}")
        return

    print("interactive chat. blank line or 'quit' to exit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        print(f"Bot: {answer(q)}")


if __name__ == "__main__":
    main()
