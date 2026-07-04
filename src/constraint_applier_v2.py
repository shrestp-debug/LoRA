"""
constraint_applier_v2.py — Phase 3b (Global Write-Side Constraint)
==================================================================
Applies soft weight-space projection to ΔW=(B@A)*scaling at each evaluation step.

This version (V2) uses a single global direction vector (v) and applies 
a write-side (left-multiply) projection to o_proj and down_proj modules.

delta_W_safe = delta_W - lam * (v @ (v.T @ delta_W))
"""

import logging
import torch

logger = logging.getLogger(__name__)

class ConstraintApplierV2:
    def __init__(self, model, v: torch.Tensor, device: str, target_layers, initial_lambda=0.0):
        self.model = model
        self.device = device
        self.v = v.to(device, dtype=torch.float32)
        self.module_keys = [f"model.layers.{i}.self_attn.o_proj" for i in target_layers] + \
                            [f"model.layers.{i}.mlp.down_proj" for i in target_layers]
        self.lambdas = {k: initial_lambda for k in self.module_keys}

    def set_all_lambdas(self, lambda_dict):
        for k, val in lambda_dict.items():
            if k in self.lambdas:
                self.lambdas[k] = max(0.0, min(1.0, float(val)))

    def get_lambdas(self):
        return dict(self.lambdas)

    def apply_projection(self):
        n_applied = 0
        named = dict(self.model.named_modules())
        for key in self.module_keys:
            lam = self.lambdas.get(key, 0.0)
            if lam <= 0.0:
                continue
            target = next((m for name, m in named.items() if key in name and hasattr(m, "lora_A")), None)
            if target is None:
                continue
            
            with torch.no_grad():
                B = target.lora_B["default"].weight.detach().to(torch.float32)
                proj = torch.outer(self.v, self.v @ B)
                B_new = B - lam * proj
                target.lora_B["default"].weight.data.copy_(B_new.to(target.lora_B["default"].weight.dtype))
            n_applied += 1
        return n_applied