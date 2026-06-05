# MuaLLM — Project Context / Handoff

Paste this whole file into ChatGPT (or any assistant) to continue the project on
another device. It is the single source of truth for goal, architecture, the
pipeline, every bug already solved, and what to do next.

> **You (the assistant) are helping build a small from-scratch chatbot by
> distilling a local open-weight teacher model. No paid APIs. Least compute,
> highest impact. Education/teaching domain.**

---

## 1. Goal & strategy

- **Goal:** our own generative chatbot, **no API**, least compute, education
  domain.
- **Why distillation:** a tiny (~20M param) model trained from scratch on a
  small corpus produces gibberish — it is a scale problem, not a bug. Instead we
  **distill a local open-weight instruct teacher** (runs on Colab/Kaggle GPU,
  not an API) into a tiny student. Soft top-k logits transfer the teacher's
  distribution ~10–100× more sample-efficiently than hard labels.
- **Repo:** https://github.com/Abdus-Sami01/MuaLLM  (branch `main`)
- **Local path:** `F:\python_files\MuaLLM`

## 2. History

Started as `slm_qa` — encoder-only extractive QA. Pivoted to a **decoder-only
causal LM chatbot**. From-scratch pretrain on ~50MB wiki gave gibberish →
switched to the distillation plan above.

## 3. Architecture (`src/model/`)

- `decoder.py` — `Decoder`: token + positional embedding → N pre-LN
  `DecoderBlock`s → final LayerNorm. Outputs hidden states. **Supports
  `pad_id=None`** (no padding semantics; needed for the distill path).
- `block.py` — `DecoderBlock` (pre-LN, swappable attention, GELU FFN).
  `ATTENTION_REGISTRY = {softmax, linear, rwkv, mamba}`.
- `attention/` — `softmax` (causal, `tril`), `linear` (causal, cumsum),
  `rwkv` (causal; the reverse/bidirectional pass is dead code, gated off),
  `ssm/mamba2` (real `mamba-ssm` on CUDA, fallback otherwise). **All causal.**
- `heads.py` — `CausalLMHead` (weight tied to embedding) + legacy `QASpanHead`.
- Size: ~20M params at `d_model=256`, vocab 49152 (teacher's).

## 4. Tokenizer — two paths

- **Project BPE** (`src/tokenizer/train_bpe.py`, vocab 8k, `[CLS] [SEP] [PAD]
  [MASK]`) — used by the SFT path.
- **Distill path uses the TEACHER's HF tokenizer.** The student is tied to it,
  so **teacher vocab sets student embed size**: SmolLM2 49k (use this),
  Phi-3-mini 32k (smaller but 3.8B teacher = slow), Qwen2.5-0.5B 152k (too big).

## 5. Pipeline (recommended = logit-KD distillation)

1. **`src/data/gen_sft.py`** — local HF instruct teacher generates education
   Q→A pairs. Writes `data/qa/edu_seqkd.jsonl` (for BPE SFT) **and**
   `data/raw/edu_seqkd.txt` (blank-line docs → the distill corpus). bf16 on
   CUDA, `--system` prompt, `renormalize_logits`.
2. **`python -m src.train.distill cache`** — runs the teacher once, stores
   top-k logits to shards. **Injects the teacher EOS id between blank-line docs**
   so the student can learn to stop (`--no-eos-between-docs` to disable). bf16.
3. **`python -m src.train.distill train`** — trains the student with
   `L = α·CE + (1-α)·T²·KL(student‖teacher_topk)`. `pad_id=None`. **Warmup is
   capped at `total_steps//10`.**
4. **`src/infer/generate.py`** — `TokAdapter` + `build_tokenizer` load either a
   BPE checkpoint (`tokenizer_path`, `[CLS]`/`[SEP]`) or a distilled checkpoint
   (`tokenizer_id`, HF BOS/EOS). `chat.py` is a thin wrapper over it.

## 6. Commands

```bash
# 1. generate corpus from local teacher
python -m src.data.gen_sft --teacher HuggingFaceTB/SmolLM2-360M-Instruct \
    --n 8000 --batch-size 32 --max-new-tokens 220 \
    --out data/qa/edu_seqkd.jsonl --txt data/raw/edu_seqkd.txt

# 2. cache teacher top-k logits (EOS auto-injected)
python -m src.train.distill cache --corpus data/raw/edu_seqkd.txt \
    --teacher HuggingFaceTB/SmolLM2-360M-Instruct --out data/distill/logits \
    --max-len 512 --topk 20

# 3. distill the student
python -m src.train.distill train --logit-dir data/distill/logits \
    --tokenizer-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --attention softmax --d-model 256 --n-layers 6 --d-ff 1024 \
    --epochs 10 --batch-size 16 --ckpt checkpoints/edu_distill.pt

# 4. chat
python chat.py checkpoints/edu_distill.pt
# or: python -m src.infer.generate --ckpt checkpoints/edu_distill.pt --prompt "..."
```

## 7. Colab notebook

- `notebooks/colab_distill.ipynb` (built by `notebooks/_build_colab_distill.py`,
  **clone-based** so it tracks `main`).
- Open: https://colab.research.google.com/github/Abdus-Sami01/MuaLLM/blob/main/notebooks/colab_distill.ipynb
- Set Runtime → GPU (T4). Cell 2 does `git pull` on re-run. Config in cell 3.
- **Do NOT `pip install -r requirements.txt` on Colab** (see gotchas).

## 8. Gotchas already solved (do not reintroduce)

1. **Colab numpy ABI:** `requirements.txt` had `numpy<2` → it downgrades Colab's
   numpy and breaks torch ABI (`numpy.dtype size changed`). Notebook installs
   only `transformers accelerate`; repo pin relaxed to `numpy>=1.24`.
2. **fp16 sampling crash:** fp16 logits overflow → `inf/nan` → multinomial CUDA
   assert. Use **bf16** (fp32 exponent range, sampling-safe, half memory).
3. **EOS / pad collision:** when the teacher has no pad token, setting
   `pad_id=eos` made the Decoder mask EOS from attention, zero its (tied)
   embedding, and mask it in labels → student could never stop. Fix: distill
   uses `pad_id=None`; cache injects teacher EOS between docs.
4. **Word-salad = LR warmup bug:** default `warmup_steps=200` exceeded a small
   run's total steps, so the LR never ramped and the student barely trained.
   Fix: `warmup = min(warmup_steps, total_steps//10)` (now logged).
5. **GPU "underused" (1.9GB) is normal** for a 360M teacher — chase throughput
   (batch size / GPU util %), not RAM. bf16 + bigger batch + bigger teacher use
   the headroom.

## 9. Current state (2026-06-05)

- Pipeline works end-to-end: bot loads, talks, and stops.
- First full run output was word-salad → traced to the warmup bug + tiny data
  (only 62k teacher tokens). Both addressed.
- **Pending:** re-run cells 4→7 with the fixes (N_PAIRS=8000, EPOCHS=10,
  warmup capped). Verify `total_steps ≫ warmup`, loss drops, output is topical
  and semi-grammatical (not salad).
- Recent commits: `2be0a57` warmup+scale, `705aeed` bf16+batch+1.7B toggle,
  `b67ceac` colab numpy/bf16/system, `7fc5dc3` run_config.

## 10. Next steps

1. Re-run the pipeline with current fixes; confirm coherence improves.
2. If topical: scale for quality — `N_PAIRS=20000`, `D_MODEL=384`,
   `D_FF=1536`, optionally teacher `HuggingFaceTB/SmolLM2-1.7B-Instruct`
   (same 49k vocab → student size unchanged, better data).
3. **Fluent but factually wrong is expected** for a small generative LM. For
   correct answers add **RAG** (retrieve the relevant guide passage, prepend to
   the prompt) — not more pretraining.
4. Optional: attention ablation (`--attention linear|rwkv|mamba`).

## 11. ETA tiers (single T4)

| Tier | Result | Effort |
|---|---|---|
| Pipeline green | loads, talks, stops | **done** |
| Coherent narrow bot | fluent education answers | ~1 week (20k+ pairs, d=384) |
| Factual | trustworthy answers | + RAG layer |
| General ChatGPT-like | open domain | not feasible at this scale |

## 12. Constraints / rules

- No paid APIs anywhere (teacher = local open weights).
- Least compute, highest impact. Prove it works small before scaling.
- Windows local dev (no GPU, no `transformers` locally); training runs on
  Colab/Kaggle/Azure GPU.
