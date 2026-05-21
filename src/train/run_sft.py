"""Supervised Fine-Tuning (SFT) script for the Chatbot."""
import argparse
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.train_bpe import load_tokenizer
from src.model.decoder import Decoder
from src.model.heads import CausalLMHead
from src.train.train_clm import save_checkpoint


class SFTDataset(Dataset):
    def __init__(self, jsonl_paths, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        for path in jsonl_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    self.samples.append(data["text"])

        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        print(f"SFTDataset: {len(self.samples)} samples from {len(jsonl_paths)} file(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # Encode the WHOLE text in one pass. add_special_tokens=False disables
        # the tokenizer's [CLS]/[SEP] post-processor template, so it is not
        # re-applied on every call. Literal "[SEP]" inside the text is still
        # mapped to its special-token id by the tokenizer's added-token table.
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [self.cls_id] + enc.ids
        ids = ids[:self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len

        input_ids = ids[:-1]
        labels = ids[1:]
        labels = [l if l != self.pad_id else -100 for l in labels]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["data/qa/pk_teaching_qa.jsonl",
                             "data/qa/github_edu_qa.jsonl"],
                    help="one or more SFT jsonl files")
    ap.add_argument("--ckpt", required=True, help="Path to your Colab pretrained checkpoint")
    ap.add_argument("--out", default="checkpoints/chatbot_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap total optimizer steps (for quick smoke runs)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json path; overrides the one in the checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.ckpt}")
    
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt['config']
    
    tok_path = args.tokenizer or ckpt.get('tokenizer_path', 'data/processed/tokenizer.json')
    tokenizer = load_tokenizer(tok_path)
    print(f"tokenizer: {tok_path}  vocab={tokenizer.get_vocab_size()}")
    if tokenizer.get_vocab_size() != config['vocab_size']:
        raise SystemExit(
            f"vocab mismatch: tokenizer={tokenizer.get_vocab_size()} "
            f"checkpoint={config['vocab_size']}. Pass the matching --tokenizer."
        )
    
    ds = SFTDataset(args.data, tokenizer, max_len=config['max_len'])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    
    decoder = Decoder(
        vocab_size=config['vocab_size'], d_model=config['d_model'], n_heads=config['n_heads'],
        n_layers=config['n_layers'], d_ff=config['d_ff'], max_len=config['max_len'],
        attention=config['attention'], pad_id=config['pad_id'],
    )
    decoder.load_state_dict(ckpt['decoder'])
    decoder.to(device)
    
    clm_head = CausalLMHead(config['d_model'], config['vocab_size'], tied_weight=decoder.embed.tok.weight)
    clm_head.load_state_dict(ckpt['clm_head'])
    clm_head.to(device)
    
    decoder.train()
    clm_head.train()
    
    # Lower learning rate for fine-tuning!
    params = list(decoder.parameters()) + list(clm_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    
    print(f"Starting Fine-Tuning on {len(ds)} QA pairs for {args.epochs} epochs...")
    step = 0
    stop = False
    for ep in range(args.epochs):
        if stop:
            break
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            hidden = decoder(input_ids)
            logits = clm_head(hidden)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % 5 == 0:
                print(f"[Epoch {ep}] Step {step} | Loss: {loss.item():.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
            
    # Save config + tokenizer_path so inference can rebuild the model.
    # opt state omitted to keep the file small (not needed for inference).
    save_checkpoint(args.out, decoder, clm_head, opt=None,
                    meta={"finetuned": True, "epochs": args.epochs,
                          "config": config, "tokenizer_path": tok_path})
    print(f"Saved Fine-Tuned Chatbot to {args.out}!")

if __name__ == "__main__":
    main()
