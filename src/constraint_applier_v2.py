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
        # NOTE: o_proj/down_proj have NO LoRA adapters in the current pipeline —
        # they are base-model matrices. To make them constrainable, target_modules
        # in build_lora_model MUST include o_proj and down_proj, and this function
        # must project each LoRA-derived delta_W the same way ConstraintApplier
        # already does for q/v, but write-side: delta_W_safe = delta_W - lam * torch.outer(v, v @ delta_W)
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
                A = target.lora_A["default"].weight.detach().to(torch.float32)
                B = target.lora_B["default"].weight.detach().to(torch.float32)
                scaling = target.scaling.get("default", 1.0)
                delta_W = (B @ A) * scaling
                
                # Write-side projection using the global vector v
                proj = torch.outer(self.v, self.v @ delta_W)
                delta_W_safe = delta_W - lam * proj
                
                target_W = delta_W_safe / scaling
                r = A.shape[0]
                U_svd, S_svd, Vh_svd = torch.linalg.svd(target_W, full_matrices=False)
                new_B = U_svd[:, :r] * torch.sqrt(S_svd[:r].clamp(min=0.0)).unsqueeze(0)
                new_A = torch.sqrt(S_svd[:r].clamp(min=0.0)).unsqueeze(1) * Vh_svd[:r, :]

                # --- DIAGNOSTIC: log discontinuity magnitude vs lambda change ---
                delta_A_norm = torch.norm(new_A.to(A.dtype) - A).item()
                delta_B_norm = torch.norm(new_B.to(B.dtype) - B).item()
                if not hasattr(self, "_svd_jump_log"):
                    self._svd_jump_log = []
                self._svd_jump_log.append({
                    "key": key, "lambda": lam,
                    "delta_A_norm": delta_A_norm, "delta_B_norm": delta_B_norm,
                })
                # -----------------------------------------------------------

                target.lora_B["default"].weight.data.copy_(new_B.to(target.lora_B["default"].weight.dtype))
                target.lora_A["default"].weight.data.copy_(new_A.to(target.lora_A["default"].weight.dtype))
            n_applied += 1
        return n_applied

    def get_svd_jump_log(self):
        return getattr(self, "_svd_jump_log", [])