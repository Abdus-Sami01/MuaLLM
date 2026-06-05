"""Generate notebooks/colab_distill.ipynb - end-to-end distillation on GPU.

Run once:  python notebooks/_build_colab_distill.py

Unlike the older inline notebooks (colab_pretrain*/colab_sft), this one CLONES
the repo and runs the real `python -m src...` CLIs, so it always matches the
codebase. Push your local changes to `main` before running it in Colab.

Pipeline (no API, no gibberish-prone from-scratch pretrain):
  local instruct teacher -> gen_sft corpus -> distill cache -> distill train
  -> generate test -> download checkpoint.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "colab_distill.ipynb"
REPO = "https://github.com/Abdus-Sami01/MuallM.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


cells = []

cells.append(md(r"""# MuaLLM - End-to-End Distillation Chatbot (GPU)

Builds a small chatbot by **distilling a local open-weight instruct teacher**
into a tiny student. No API. No from-scratch pretrain (that path produces
gibberish at this scale).

**Pipeline**
1. `gen_sft` - teacher generates ~N education Q->A pairs (a corpus)
2. `distill cache` - store the teacher's top-k logits (the EOS token is
   auto-injected between docs so the student learns to **stop**)
3. `distill train` - train the tiny student with KL + CE
4. generate test -> download checkpoint

**Before you run this notebook**, push your local repo so the clone is current:
```
git add -A && git commit -m "distill pipeline" && git push
```
Then in Colab: Runtime -> Change runtime type -> **GPU (T4)**, and run cells
top to bottom.

---
"""))

cells.append(md("## 1. GPU check"))
cells.append(code(r"""import sys, torch
print('python:', sys.version.split()[0])
print('torch :', torch.__version__)
print('cuda  :', torch.cuda.is_available())
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print('device:', d.name, '|', round(d.total_memory / 1e9, 1), 'GB')
else:
    print('WARNING: no GPU. Set Runtime -> Change runtime type -> GPU (T4).')
"""))

cells.append(md(r"""## 2. Clone repo + install deps

Clones `main` and installs `transformers` (the teacher) on top of the repo's
`requirements.txt`. Re-running is safe (skips clone if the dir exists)."""))
cells.append(code(r"""import os
REPO = "https://github.com/Abdus-Sami01/MuallM.git"
if not os.path.isdir("MuallM"):
    !git clone {REPO}
%cd MuallM
!git pull   # pick up re-pushed fixes on re-run
!git log --oneline -1
# Colab already ships a matched torch + numpy (2.x). Do NOT `pip install -r
# requirements.txt` here: its numpy<2 pin downgrades numpy and breaks Colab's
# torch ABI ("numpy.dtype size changed"). Only the teacher deps are missing.
!pip install -q "transformers>=4.44" accelerate
"""))

cells.append(md(r"""## 3. Config

Defaults = a fast **smoke run** (finishes in minutes on a T4) that already
shows real fluency + clean stopping. Scale the commented values for a coherent
bot."""))
cells.append(code(r"""TEACHER     = "HuggingFaceTB/SmolLM2-360M-Instruct"  # 49k vocab, fast cache
N_PAIRS     = 3000      # -> 20000+ for a genuinely coherent bot
GEN_BATCH   = 16
MAX_NEW     = 160       # teacher answer length (tokens)
MAX_LEN     = 512       # distill block length
TOPK        = 20
ATTENTION   = "softmax" # softmax | linear | rwkv | mamba
D_MODEL     = 256       # -> 384 for tier-2 quality
N_LAYERS    = 6
D_FF        = 1024      # -> 1536 with d_model=384
EPOCHS      = 3
TRAIN_BATCH = 16
CKPT        = "checkpoints/edu_distill.pt"
print("teacher:", TEACHER, "| pairs:", N_PAIRS, "| student d_model:", D_MODEL)
"""))

cells.append(md(r"""## 4. Generate the corpus from the local teacher

Writes `edu_seqkd.jsonl` (for BPE SFT, unused here) and `edu_seqkd.txt`
(blank-line docs -> the distill corpus)."""))
cells.append(code(r"""!python -m src.data.gen_sft --teacher {TEACHER} --n {N_PAIRS} \
    --batch-size {GEN_BATCH} --max-new-tokens {MAX_NEW} \
    --out data/qa/edu_seqkd.jsonl --txt data/raw/edu_seqkd.txt
print("\n--- corpus preview ---")
!head -c 800 data/raw/edu_seqkd.txt
"""))

cells.append(md(r"""## 5. Cache the teacher's top-k logits

One-shot. EOS is auto-injected between blank-line docs (disable with
`--no-eos-between-docs`)."""))
cells.append(code(r"""!python -m src.train.distill cache --corpus data/raw/edu_seqkd.txt \
    --teacher {TEACHER} --out data/distill/logits \
    --max-len {MAX_LEN} --topk {TOPK}
"""))

cells.append(md(r"""## 6. Distill the student

Student shares the teacher tokenizer; `pad_id=None` so it can learn to emit
EOS. Checkpoint stores `config` + `tokenizer_id` for inference."""))
cells.append(code(r"""!python -m src.train.distill train --logit-dir data/distill/logits \
    --tokenizer-id {TEACHER} --attention {ATTENTION} \
    --d-model {D_MODEL} --n-layers {N_LAYERS} --d-ff {D_FF} \
    --epochs {EPOCHS} --batch-size {TRAIN_BATCH} --ckpt {CKPT}
"""))

cells.append(md(r"""## 7. Talk to it

Loads the checkpoint once via the repo's inference lib (HF-tokenizer flavor is
detected automatically) and answers a few questions. It should produce fluent
sentences and **stop on its own** (EOS)."""))
cells.append(code(r"""import torch
from src.infer.generate import load_model, build_tokenizer, generate

device = "cuda" if torch.cuda.is_available() else "cpu"
decoder, clm, config, ck = load_model(CKPT, device)
tok = build_tokenizer(ck)
print(f"loaded {CKPT}  tok={tok.kind}  vocab={config['vocab_size']}  device={device}\n")

for q in ["What is formative assessment?",
          "Give two classroom management tips.",
          "How does spaced repetition help learning?"]:
    ans = generate(decoder, clm, tok, f"User: {q}\nBot:",
                   max_new_tokens=120, temperature=0.8, top_k=40, device=device)
    print("Q:", q)
    print("A:", ans)
    print("-" * 70)
"""))

cells.append(md("## 8. Save the checkpoint"))
cells.append(code(r"""from google.colab import files
files.download(CKPT)

# Or persist to Google Drive instead:
# from google.colab import drive; drive.mount('/content/drive')
# import shutil; shutil.copy(CKPT, '/content/drive/MyDrive/edu_distill.pt')
"""))

cells.append(md(r"""## 9. Next steps

- **Fluent but generic?** Scale: `N_PAIRS=20000`, `D_MODEL=384`, `D_FF=1536`.
  Re-run cells 4-6.
- **Fluent but factually wrong?** That is expected for a small generative LM.
  Add a **RAG** layer (retrieve the relevant guide passage, prepend it to the
  prompt) rather than training more.
- **Try the efficient attentions** for the ablation: set `ATTENTION` to
  `linear` / `rwkv` / `mamba` (mamba needs `pip install mamba-ssm`, CUDA only).
- Chat locally after download: `python chat.py edu_distill.pt`.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}  ({len(cells)} cells)")
