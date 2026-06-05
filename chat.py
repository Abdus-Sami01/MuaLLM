"""Interactive chatbot CLI. Thin wrapper over src.infer.generate.

Loads a trained checkpoint (SFT or distilled — tokenizer flavor is detected
automatically) and runs an interactive loop.

  python chat.py                       # uses default checkpoint below
  python chat.py checkpoints/edu_distill.pt
"""
import sys
from pathlib import Path

import torch

from src.infer.generate import load_model, build_tokenizer, generate

DEFAULT_CKPT = "checkpoints/chatbot_finetuned.pt"


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    if not Path(ckpt_path).exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        print("Pass a path: python chat.py <checkpoint.pt>")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, clm, config, ck = load_model(ckpt_path, device)
    tok = build_tokenizer(ck)
    print(f"\n=== Chatbot Ready ({tok.kind} tokenizer, {device}) ===")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        reply = generate(decoder, clm, tok, f"User: {user_input}\nBot:",
                         max_new_tokens=100, temperature=0.8, device=device)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    main()
