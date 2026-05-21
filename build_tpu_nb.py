import json
from pathlib import Path

nb_path = Path("notebooks/colab_pretrain_clm.ipynb")
if not nb_path.exists():
    print(f"Error: {nb_path} not found.")
    exit(1)

nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # 1. Device setup (Ensure we only target the actual environment check cell)
        if "torch.cuda.is_available()" in src and "import sys, os, platform" in src:
            src = """import os
# Enable TPU bfloat16 automatically
os.environ["XLA_USE_BF16"] = "1" 

import sys, platform
import torch
import torch_xla
import torch_xla.core.xla_model as xm

print('python   :', sys.version.split()[0])
print('torch    :', torch.__version__)
print('torch_xla:', torch_xla.__version__)

DEVICE = torch_xla.device()
print('DEVICE   :', DEVICE)
"""
            cell['source'] = [src]
            
        # 2. Train loop
        elif "def train_clm_amp" in src:
            src = """import time, math
import torch
import torch_xla.core.xla_model as xm

def make_lr_scheduler(opt, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def train_clm_tpu(decoder, clm_head, dataset, *, epochs=1, batch_size=32,
                  lr=3e-4, weight_decay=0.01, grad_clip=1.0,
                  device="cpu", log_every=20, num_workers=2,
                  max_steps=None, warmup_steps=50, grad_accum=1,
                  min_lr_ratio=0.1):
    decoder.train(); clm_head.train()
    decoder.to(device); clm_head.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True)
    
    seen, params = set(), []
    for p in list(decoder.parameters()) + list(clm_head.parameters()):
        if id(p) in seen: continue
        seen.add(id(p)); params.append(p)
        
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    steps_per_epoch = len(loader) // grad_accum
    total_opt_steps = max_steps if max_steps else steps_per_epoch * epochs
    sched = make_lr_scheduler(opt, warmup_steps, total_opt_steps, min_lr_ratio)

    losses = []
    step = 0; opt_step = 0; t0 = time.time()
    opt.zero_grad()
    
    for ep in range(epochs):
        for micro_i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            hidden = decoder(input_ids)
            logits = clm_head(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accum
            loss.backward()
            
            # Use xm.mark_step() for printing accurate loss values in XLA
            xm.mark_step()
            losses.append(loss.item() * grad_accum)

            do_step = ((micro_i + 1) % grad_accum == 0)
            if do_step:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                # xm.optimizer_step takes care of step and mark_step
                xm.optimizer_step(opt)
                sched.step()
                opt.zero_grad()
                opt_step += 1

                if opt_step % log_every == 0:
                    dt = time.time() - t0
                    eff_bs = batch_size * grad_accum
                    tps = (opt_step * eff_bs * dataset.max_len) / max(dt, 1e-3)
                    cur_lr = sched.get_last_lr()[0]
                    cur_loss = sum(losses[-grad_accum:]) / grad_accum
                    print(f"[ep {ep} opt_step {opt_step:5d}] "
                          f"loss={cur_loss:.4f}  ppl={math.exp(min(cur_loss, 20)):.1f}  "
                          f"lr={cur_lr:.2e}  tok/s={tps:.0f}  elapsed={dt:.1f}s")

                if max_steps and opt_step >= max_steps:
                    return losses
            step += 1
    return losses
"""
            cell['source'] = [src]
            
        # 3. Config changes
        elif "ATTENTION    =" in src:
            src = src.replace('WARMUP_STEPS = 500', 'WARMUP_STEPS = 50')
            src = src.replace('LR           = 5e-4', 'LR           = 1e-3')
            src = src.replace('EPOCHS       = 3', 'EPOCHS       = 10')
            cell['source'] = [src]
            
        # 4. Call changes
        elif "train_clm_amp(" in src:
            src = src.replace('train_clm_amp', 'train_clm_tpu')
            cell['source'] = [src]

Path("notebooks/colab_pretrain_tpu.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("TPU notebook successfully generated!")
