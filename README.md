<div align="center">

# MuaLLM

**A ~20M-parameter language model built from scratch — tokenizer, attention kernels, training loop and all — and taught to chat by distilling a local open-weight teacher. No paid APIs, no big GPUs.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Teacher](https://img.shields.io/badge/teacher-SmolLM2--360M--Instruct-FFD21E.svg)](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
[![Compute](https://img.shields.io/badge/compute-CPU%20%2F%20free%20Colab-22c55e.svg)]()

</div>

---

## Why this exists

Training a tiny model from scratch on a small corpus produces gibberish — that's a **scale problem, not a bug**. MuaLLM's answer is **logit-level knowledge distillation**: run a local open-weight instruct teacher once, cache its top-k logits, and train the tiny student against the teacher's full distribution instead of hard labels. Soft targets transfer knowledge ~10–100× more sample-efficiently, which is what makes a from-scratch ~20M model usable on hobbyist compute.

The second experiment baked in from the start: **the attention block is swappable by config**, so subquadratic variants can be benchmarked head-to-head on identical everything-else.

| Variant | Complexity | Notes |
|---|---|---|
| Softmax | `O(n²)` | causal scaled dot-product baseline |
| Linear attention | `O(n·d²)` | Katharopoulos et al. 2020, causal cumsum |
| RWKV time-mix | `O(n·d)` | linear recurrence with token-shift |
| Mamba-2 | `O(n)` | real `mamba-ssm` kernel on CUDA, fallback elsewhere |

## Architecture

- **Decoder-only, pre-LN**, ~20M params at `d_model=256`; token+positional embeddings → N `DecoderBlock`s (swappable attention + GELU FFN) → tied `CausalLMHead`
- **Tokenizer, two paths**: a from-scratch 8k BPE (trained locally with `tokenizers`) for the SFT path, or the teacher's HF tokenizer for the distillation path (student embedding size follows teacher vocab)
- **Distillation loss**: `L = α·CE + (1−α)·T²·KL(student ∥ teacher-top-k)`

## Pipeline

```bash
# 1. Local teacher generates an education-domain corpus (no API calls)
python -m src.data.gen_sft --teacher HuggingFaceTB/SmolLM2-360M-Instruct \
    --n 8000 --out data/qa/edu_seqkd.jsonl --txt data/raw/edu_seqkd.txt

# 2. Cache teacher top-k logits to shards (teacher runs exactly once)
python -m src.train.distill cache --corpus data/raw/edu_seqkd.txt \
    --teacher HuggingFaceTB/SmolLM2-360M-Instruct --out data/distill/logits \
    --max-len 512 --topk 20

# 3. Train the student against the cached distribution
python -m src.train.distill train

# 4. Chat with it
python chat.py
```

Sanity check the whole stack in under a minute:

```bash
python -m src.train.smoke_test   # decreasing loss on a toy corpus
```

## Layout

```
src/
  tokenizer/   from-scratch BPE training
  data/        corpus extraction, chunking, teacher generation
  model/
    attention/ softmax.py, linear.py, rwkv.py, ssm/ (mamba2)
    block.py, decoder.py, heads.py
  train/       distill.py (cache + train), pretrain_mlm.py, finetune_qa.py
  infer/       generate.py (TokAdapter loads either tokenizer path)
configs/       base.yaml + one config per attention variant
```

## Project history

Started life as `slm_qa`, an encoder-only extractive QA model (BPE → MLM pretrain on a ~50 MB Wikipedia education subset → span head), trained entirely on CPU. The from-scratch generative attempt at this scale confirmed the gibberish wall, which motivated the pivot to distillation — the negative result is part of the point. The legacy QA path (`QASpanHead`, MLM pretraining) is still in the repo and still runs.

## Hardware

Everything targets **CPU or free-tier GPUs** (Colab/Kaggle T4): teacher generation and logit caching in bf16 on a free GPU, student training on CPU if needed. Open-source stack only — `torch`, `tokenizers`, `datasets`; no API wrappers anywhere.
