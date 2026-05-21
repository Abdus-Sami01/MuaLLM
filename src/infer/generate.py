"""Autoregressive generation / chat with a fine-tuned causal LM checkpoint.

Usage:
  # single question
  python -m src.infer.generate --ckpt checkpoints/chatbot_finetuned.pt \
      --prompt "What is the Single National Curriculum?"

  # interactive chat loop
  python -m src.infer.generate --ckpt checkpoints/chatbot_finetuned.pt
"""
import argparse

import torch
import torch.nn.functional as F

from src.tokenizer.train_bpe import load_tokenizer
from src.model.decoder import Decoder
from src.model.heads import CausalLMHead


@torch.no_grad()
def generate(decoder, clm_head, tokenizer, prompt, *, max_new_tokens=120,
             temperature=0.8, top_k=40, device="cpu"):
    """Greedy/top-k sampled continuation. Stops at [SEP]."""
    decoder.eval()
    clm_head.eval()
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    max_len = decoder.embed.max_len

    ids = [cls_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    generated = []
    for _ in range(max_new_tokens):
        window = ids[-max_len:]
        inp = torch.tensor([window], dtype=torch.long, device=device)
        hidden = decoder(inp)
        logits = clm_head(hidden)[0, -1, :]
        logits = logits / max(temperature, 1e-5)
        if top_k:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == sep_id:
            break
        ids.append(nxt)
        generated.append(nxt)
    return tokenizer.decode(generated)


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    config = ck.get("config") or ck.get("meta", {}).get("config")
    if config is None:
        raise SystemExit(
            "checkpoint has no 'config'. Re-run SFT with the fixed run_sft.py "
            "(it now stores config + tokenizer_path in meta)."
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
    tok_path = ck.get("tokenizer_path") or ck.get("meta", {}).get("tokenizer_path")
    return decoder, clm, config, tok_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="override tokenizer path (else read from checkpoint)")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--prompt", default=None,
                    help="single question; if omitted, interactive chat loop")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, clm, config, tok_path = load_model(args.ckpt, device)
    tok_path = args.tokenizer or tok_path
    if not tok_path:
        raise SystemExit("no tokenizer path in checkpoint; pass --tokenizer")
    tokenizer = load_tokenizer(tok_path)
    print(f"loaded {args.ckpt}  attention={config['attention']}  device={device}")

    def answer(question):
        prompt = f"User: {question}\nBot:"
        return generate(decoder, clm, tokenizer, prompt,
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
