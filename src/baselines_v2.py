"""
baselines_v2.py — Phase 3b (Global, Write-Side Safety Constraints)
====================================================================
IMPORTANT NAMING NOTE:
These are NOT reproductions of the SafeLoRA/SaLoRA papers. Both papers
specify per-layer, read-side (q_proj/v_proj) directions derived from
harmful-prompt activations. A causal ablation study on this model found
that substrate does not gate refusal — refusal is instead concentrated
in the write-side residual-stream contributions (o_proj/down_proj),
via a single global direction (diff-of-means, layer 14).

The functions below borrow the *mechanism* each paper uses (periodic
weight-space projection for SafeLoRA-style; per-forward-pass output
projection for the SaLoRA-style hook) and apply it to the causally
verified substrate instead. They are mechanism-adapted baselines, not
paper replications. The faithful replications (run_salora.py, with
per-layer read-side directions, task-init, and reparameterization)
remain unmodified in the v1 pipeline and are reported separately.
"""

import torch
import contextlib
import logging
import random

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def safelora_eval_context(model, v):
    """
    SafeLoRA-adapted-v2: periodic weight-space projection, write-side.

    Formula:
      ΔW = (B @ A) * scaling
      ΔW_safe = ΔW - torch.outer(v, v @ ΔW)
      W_eval = W_base + ΔW_safe
    """
    logger.info("[SafeLoRA-adapted-v2] Entering periodic evaluation context...")

    deltas = {}
    norm_ratios = []

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            adapter_name = "default"
            if adapter_name not in module.lora_A:
                continue

            A = module.lora_A[adapter_name].weight  # (r, d_in)
            B = module.lora_B[adapter_name].weight  # (d_out, r)
            scaling = module.scaling[adapter_name]

            delta_W = (B @ A) * scaling  # (d_out, d_in)
            v_dev = v.to(device=delta_W.device, dtype=delta_W.dtype)

            proj = torch.outer(v_dev, v_dev @ delta_W)
            delta_W_safe = delta_W - proj

            norm_before = torch.norm(delta_W).item()
            norm_after = torch.norm(delta_W_safe).item()
            ratio = norm_after / norm_before if norm_before > 0 else 0.0
            norm_ratios.append(ratio)

            deltas[name] = delta_W_safe
            module.base_layer.weight.data += delta_W_safe

    avg_ratio = sum(norm_ratios) / len(norm_ratios) if norm_ratios else 0.0
    logger.info(f"[SafeLoRA-adapted-v2] Projected weight norm ratio ||ΔW_safe|| / ||ΔW||: {avg_ratio:.4f}")

    with model.disable_adapter():
        yield

    for name, module in model.named_modules():
        if name in deltas:
            module.base_layer.weight.data -= deltas[name]

    logger.info("[SafeLoRA-adapted-v2] Exited evaluation context, restored base weights.")


def apply_write_side_hook_constraint(model, v):
    """
    SaLoRA-mechanism-inspired-v2: continuous (every-forward-pass) output
    projection hook, applied write-side against the global safety direction.

    This borrows SaLoRA's per-forward-pass hook mechanism but does NOT
    reproduce SaLoRA's per-layer read-side directions, task-specific
    adapter initialization, or weight reparameterization. Do not report
    this as "SaLoRA" in the paper — the faithful replication is
    run_salora.py (v1), reported separately.

    Formula:
      lora_output_safe = lora_output - (lora_output @ v) * v
    """
    logger.info("[write-side-hook-v2] Applying fixed forward post-hooks...")
    hooks = []

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            adapter_name = "default"
            if adapter_name not in module.lora_A:
                continue

            lora_B = module.lora_B[adapter_name]

            def post_hook(mod, args, output, v_base=v):
                v_dev = v_base.to(device=output.device, dtype=output.dtype)
                proj = torch.einsum('bsd,d->bs', output, v_dev).unsqueeze(-1) * v_dev
                out_proj = output - proj

                if not hasattr(mod, "hook_stats"):
                    mod.hook_stats = []
                if random.random() < 0.001:
                    norm_out = torch.norm(output).item()
                    norm_out_proj = torch.norm(out_proj).item()
                    mod.hook_stats.append({
                        "original_norm": norm_out,
                        "projected_norm": norm_out_proj,
                        "ratio": norm_out_proj / norm_out if norm_out > 0 else 0
                    })

                return out_proj

            hook_handle = lora_B.register_forward_hook(post_hook)
            hooks.append(hook_handle)

    logger.info(f"[write-side-hook-v2] Successfully applied {len(hooks)} output projection hooks.")
    return hooks


def log_write_side_hook_stats(model):
    all_stats = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_B"):
            adapter_name = "default"
            if adapter_name in module.lora_B and hasattr(module.lora_B[adapter_name], "hook_stats"):
                all_stats.extend(module.lora_B[adapter_name].hook_stats)

    if not all_stats:
        return

    avg_ratio = sum(s["ratio"] for s in all_stats) / len(all_stats)
    logger.info(f"[write-side-hook-v2 verification] Average norm ratio after projection: {avg_ratio:.6f}")


def load_global_safety_direction(models_dir):
    import torch
    from pathlib import Path

    path = Path(models_dir) / "global_refusal_direction_layer14.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the Phase 3b ablation/extraction study first.")

    v = torch.load(path, map_location="cpu")
    return v / v.norm()