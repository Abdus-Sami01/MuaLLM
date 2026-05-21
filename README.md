# slm_qa

Small Language Model for extractive QA on the teaching / education domain.
Semester project. Custom subquadratic attention variants. CPU-trainable.

## Goal

Given a question and a passage from a teacher profession guide (Wikipedia
education corpus + similar), predict the answer span inside the passage.

Not a chatbot. Not generative. Extractive QA only.

## Architecture

Encoder-only, ~8M parameters, swappable attention block.

```
Input: [CLS] question [SEP] context [SEP]
   |
   v  Embedding(vocab=8k, d=256) + positional
   |
   v  N x EncoderBlock(attention=<variant>, ffn=1024)
   |
   v  SpanHead -> (start_logits, end_logits)
```

## Attention variants implemented

1. **Softmax** (baseline, `O(n^2)`) — vanilla scaled dot-product
2. **Linear attention** (Katharopoulos et al. 2020, `O(n d^2)`) — `phi(Q)(phi(K)^T V)`
3. **RWKV time-mix** (`O(n d)`) — linear recurrence with token-shift

All three share the same encoder skeleton. Swap by config.

## Training stages

1. **Tokenizer**: train BPE (vocab 8k) on raw corpus.
2. **MLM pretrain**: masked language modeling on ~50 MB Wikipedia education
   subset. CPU, 2-3 days, 1-2 epochs.
3. **QA fine-tune**: extractive span head on ~2000 synthetic QA pairs
   (manual + rule-based).
4. **Eval**: Exact Match, F1, latency, peak memory. Three variants compared.

## Hardware target

- Training: CPU on local Windows PC (8-16 core, 16-32 GB RAM)
- Optional: Kaggle T4 GPU for ablation runs (30 hr/week free tier)

## Stack

Open source only. No API wrappers.

- `torch` — framework
- `tokenizers` — BPE training (local, not API)
- `datasets` — load Wikipedia dump locally
- `pdfplumber`, `beautifulsoup4` — text extraction
- `numpy`, `matplotlib`, `tqdm`

## Layout

```
slm_qa/
  data/{raw,processed,qa}/
  src/
    tokenizer/   BPE training
    data/        extract.py, chunk.py, qa_gen.py
    model/
      attention/ softmax.py, linear.py, rwkv.py
      block.py, encoder.py, heads.py
    train/       pretrain_mlm.py, finetune_qa.py
    eval/        metrics.py, benchmark.py
  configs/       base.yaml, variant_*.yaml
  scripts/       download_wiki.ps1, run_*.ps1
  tests/         unit tests
  notebooks/     EDA + ablation plots
```

## Quick smoke test

```
python -m src.train.smoke_test
```

Should print decreasing loss on a toy corpus in under a minute.

## Status

- [x] Scaffold
- [ ] Attention variants
- [ ] Encoder + heads
- [ ] Tokenizer + data pipeline
- [ ] MLM pretrain
- [ ] QA fine-tune
- [ ] Eval + ablation
