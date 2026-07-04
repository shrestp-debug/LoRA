"""
check_natural_alignment.py
Computes mean_alignment on an EXISTING (unconstrained) SafeLoRA-v2 or
vanilla-v2 checkpoint, to see what natural drift alignment with v looks
like before you commit to the persistence fix.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from src.baselines_v2 import load_global_safety_direction

def compute_alignment(model, v, target_layers):
    named = dict(model.named_modules())
    keys = [f"model.layers.{i}.self_attn.o_proj" for i in target_layers] + \
           [f"model.layers.{i}.mlp.down_proj" for i in target_layers]
    out = {}
    for key in keys:
        target = next((m for name, m in named.items() if key in name and hasattr(m, "lora_A")), None)
        if target is None:
            out[key] = 0.0
            continue
        A = target.lora_A["default"].weight.detach().to(torch.float32)
        B = target.lora_B["default"].weight.detach().to(torch.float32)
        scaling = target.scaling.get("default", 1.0)
        delta_W = (B @ A) * scaling
        U, _, _ = torch.linalg.svd(delta_W, full_matrices=False)
        top_left = U[:, 0]
        v_dev = v.to(dtype=torch.float32, device=top_left.device)
        out[key] = torch.abs(torch.dot(top_left, v_dev)).item()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True, help="e.g. models/safelora_v2_alpaca_seed42/checkpoint-2000")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = load_global_safety_direction(args.models_dir)

    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False).to(device)

    alignments = compute_alignment(model, v, range(28))
    vals = list(alignments.values())
    print(f"mean_alignment (natural, unconstrained) = {sum(vals)/len(vals):.6f}")
    top5 = sorted(alignments.items(), key=lambda x: x[1], reverse=True)[:5]
    for k, a in top5:
        print(f"  {k}: {a:.6f}")
