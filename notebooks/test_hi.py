import sys
import torch
from tokenizers import Tokenizer

sys.path.append("f:/python_files/slm_qa")
from src.model.encoder import Encoder
from src.model.heads import MLMHead

def main():
    device = torch.device("cpu")
    ckpt_path = "f:/python_files/slm_qa/notebooks/pretrain_softmax_1779193339.pt"
    tok_path = "f:/python_files/slm_qa/notebooks/tokenizer.json"

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tok_path)
    
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    
    # ensure pad_id is correctly set
    config['pad_id'] = tokenizer.token_to_id("[PAD]")
    
    encoder = Encoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        attention=config['attention'],
        pad_id=config['pad_id'],
    )
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    mlm_head = MLMHead(d_model=config['d_model'], vocab_size=config['vocab_size'], tied_weight=encoder.embed.tok.weight)
    mlm_head.load_state_dict(ckpt['mlm_head'])
    mlm_head.eval()

    text = "[CLS] hi [MASK] [SEP]"
    print(f"Input text: '{text}'")
    
    # Encode manually to handle special tokens if needed
    # BPE tokenizer from tokenizers handles it if tokens exist
    ids = []
    for word in text.split():
        if word in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]:
            ids.append(tokenizer.token_to_id(word))
        else:
            ids.extend(tokenizer.encode(word).ids)

    print(f"Token IDs: {ids}")
    mask_idx = ids.index(tokenizer.token_to_id("[MASK]"))
    
    input_ids = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        hidden = encoder(input_ids)
        logits = mlm_head(hidden)
    
    mask_logits = logits[0, mask_idx, :]
    probs = torch.softmax(mask_logits, dim=-1)
    top_k = torch.topk(probs, 5)
    
    print("\nTop 5 predictions for [MASK]:")
    for i in range(5):
        tok_id = top_k.indices[i].item()
        prob = top_k.values[i].item()
        tok_str = tokenizer.id_to_token(tok_id)
        safe_tok = tok_str.encode("utf-8", "ignore").decode("utf-8")
        # On Windows cmd it might still crash, so replacing \u0120 with an underscore
        safe_tok = tok_str.replace('\u0120', ' ')
        print(f"{i+1}: '{safe_tok}' (prob: {prob:.4f})".encode('ascii', 'backslashreplace').decode('ascii'))

if __name__ == "__main__":
    main()
