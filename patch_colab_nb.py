import re
from pathlib import Path

path = Path("notebooks/_build_colab_nb.py")
content = path.read_text()

# Output name
content = content.replace('colab_pretrain.ipynb', 'colab_pretrain_clm.ipynb')
content = content.replace('NB_PATH = Path(__file__).parent / "colab_pretrain.ipynb"', 'NB_PATH = Path(__file__).parent / "colab_pretrain_clm.ipynb"')

# Softmax Causal
softmax_causal = """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
"""
content = content.replace('        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)\n', softmax_causal)

# Linear Causal
linear_causal = """
        k_cumsum = torch.cumsum(k, dim=2)
        kv = torch.einsum("bhnd,bhne->bhnde", k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        num = torch.einsum("bhnd,bhnde->bhne", q, kv_cumsum)
        denom = torch.einsum("bhnd,bhnd->bhn", q, k_cumsum).clamp(min=self.eps)
"""
content = re.sub(
    r'        kv = torch.einsum\("bhnd,bhne->bhde", k, v\)\n        k_sum = k.sum\(dim=2\)\n        num = torch.einsum\("bhnd,bhde->bhne", q, kv\)\n        denom = torch.einsum\("bhnd,bhd->bhn", q, k_sum\)\.clamp\(min=self\.eps\)\n',
    linear_causal,
    content
)

# RWKV Causal
content = content.replace('bidirectional=True', 'bidirectional=False')

# Architecture renaming
content = content.replace('EncoderBlock', 'DecoderBlock')
content = content.replace('Encoder', 'Decoder')
content = content.replace('encoder', 'decoder')
content = content.replace('MLMHead', 'CausalLMHead')
content = content.replace('mlm_head', 'clm_head')
content = content.replace('mlm', 'clm')
content = content.replace('MLMDataset', 'CLMDataset')
content = content.replace('train_mlm_amp', 'train_clm_amp')

# Dataset logic
clm_dataset = """
class CLMDataset(Dataset):
    def __init__(self, chunks, vocab_size, mask_id, pad_id, cls_id, sep_id,
                 special_ids=None, mask_prob=0.15, max_len=256):
        self.chunks = chunks
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.cls_id, self.sep_id = cls_id, sep_id
        self.special_ids = set(special_ids or [])
        self.max_len = max_len

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        body = list(self.chunks[idx])[: self.max_len - 2]
        seq = [self.cls_id] + body + [self.sep_id]
        
        pad_len = self.max_len - len(seq)
        if pad_len > 0:
            seq = seq + [self.pad_id] * pad_len
            
        input_ids = seq[:-1]
        labels = seq[1:]
        labels = [l if l != self.pad_id else -100 for l in labels]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
"""
content = re.sub(
    r'class CLMDataset\(Dataset\):.*?return \{.*?"labels": torch.tensor\(labels, dtype=torch.long\),\n        \}\n',
    clm_dataset,
    content,
    flags=re.DOTALL
)

# Fix dataset instantiation
content = content.replace('ds = CLMDataset(chunks, vocab_size, mask_id, pad_id, cls_id, sep_id', 'ds = CLMDataset(chunks, vocab_size, mask_id, pad_id, cls_id, sep_id')

Path("notebooks/_build_colab_nb_clm.py").write_text(content)
print("done")
