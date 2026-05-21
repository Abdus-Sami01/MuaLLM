"""Interactive generation loop for the Causal Language Model."""
import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

from src.model.decoder import Decoder
from src.model.heads import CausalLMHead


def load_model(ckpt_path, device="cpu"):
    if not Path(ckpt_path).exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please upload the new checkpoint from Colab and update the path!")
        sys.exit(1)

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        print("Config not found, loading from base checkpoint...")
        base_ckpt = torch.load("notebooks/pretrain_softmax_1779203247.pt", map_location=device)
        config = base_ckpt['config']
    
    # Load tokenizer
    tok_path = ckpt.get('tokenizer_path', 'data/processed/tokenizer.json')
    if not Path(tok_path).exists():
        # Fallback if path from colab doesn't exist locally
        tok_path = 'data/processed/tokenizer.json'
    
    tokenizer = Tokenizer.from_file(tok_path)
    config['pad_id'] = tokenizer.token_to_id("[PAD]")
    
    # Build model
    decoder = Decoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        attention=config['attention'],
        pad_id=config['pad_id'],
    )
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval().to(device)

    clm_head = CausalLMHead(
        d_model=config['d_model'], 
        vocab_size=config['vocab_size'], 
        tied_weight=decoder.embed.tok.weight
    )
    clm_head.load_state_dict(ckpt['clm_head'])
    clm_head.eval().to(device)
    
    return decoder, clm_head, tokenizer, config


def generate(prompt, decoder, clm_head, tokenizer, config, max_new_tokens=50, temperature=0.7, top_k=40, device="cpu"):
    # BPE tokenization
    ids = []
    for word in prompt.split():
        if word in ["[CLS]", "[SEP]", "[PAD]"]:
            ids.append(tokenizer.token_to_id(word))
        else:
            ids.extend(tokenizer.encode(word).ids)
            
    # Always start sequence with [CLS] if not provided
    cls_id = tokenizer.token_to_id("[CLS]")
    if not ids or ids[0] != cls_id:
        ids = [cls_id] + ids
        
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to max_len if needed
            input_ids = ids[-(config['max_len']):]
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            hidden = decoder(x)
            logits = clm_head(hidden)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            ids.append(next_id)
            
            # Print the new token
            token_str = tokenizer.id_to_token(next_id)
            if token_str is None:
                break
                
            # Clean up BPE space character for printing
            safe_tok = token_str.replace('\u0120', ' ')
            print(safe_tok, end="", flush=True)
            
            # Stop if we hit [SEP]
            if next_id == tokenizer.token_to_id("[SEP]"):
                break
                
    print("\n")


def main():
    CHECKPOINT_PATH = "checkpoints/chatbot_finetuned.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, clm_head, tokenizer, config = load_model(CHECKPOINT_PATH, device)
    
    print("\n=== Chatbot Ready ===")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            print("Bot: ", end="")
            
            # Format the prompt
            prompt = f"User: {user_input}\nBot:"
            generate(prompt, decoder, clm_head, tokenizer, config, max_new_tokens=100, temperature=0.8, device=device)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
