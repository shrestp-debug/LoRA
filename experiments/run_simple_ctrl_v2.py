"""
run_simple_ctrl_v2.py — Phase 3b
====================================================================
Runs Baseline 4: SimpleCtrl (dynamic threshold-based lambda) 
using write-side global constraints on o_proj and down_proj modules.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train, load_gsm8k_test,
    load_alpaca_train, load_alpaca_val,
    load_advbench,
)
from src.metrics import (
    evaluate_task_gsm8k,
    evaluate_task_alpaca,
    evaluate_safety,
    eval_perplexity,
    CLEAN_PROMPTS
)
from src.baselines_v2 import load_global_safety_direction
from experiments.train_vanilla_v2 import MaskedTrainingDataset, training_collate_fn, set_seed, build_lora_model
from src.constraint_applier_v2 import ConstraintApplierV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_subspace_alignment_v2(model, v, target_layers):
    """Computes alignment of Delta W's top right singular vector with global direction v."""
    alignments = {}
    named_modules = dict(model.named_modules())
    module_keys = [f"model.layers.{i}.self_attn.o_proj" for i in target_layers] + \
                  [f"model.layers.{i}.mlp.down_proj" for i in target_layers]
                  
    for key in module_keys:
        try:
            target_module = next((m for name, m in named_modules.items() if key in name and hasattr(m, "lora_A")), None)
            if target_module is None:
                alignments[key] = 0.0
                continue
                
            lora_A = target_module.lora_A["default"].weight.detach().to(torch.float32)  
            lora_B = target_module.lora_B["default"].weight.detach().to(torch.float32)  
            scaling = target_module.scaling.get("default", 1.0)

            delta_W = (lora_B @ lora_A) * scaling

            U, _, _ = torch.linalg.svd(delta_W, full_matrices=False)
            top_left_vector = U[:, 0]  
            v_dev = v.to(dtype=torch.float32, device=top_left_vector.device)

            alignment = torch.abs(torch.dot(top_left_vector, v_dev)).item()
            alignments[key] = alignment
        except Exception as e:
            logger.error(f"Error computing subspace alignment for {key}: {e}")
            alignments[key] = 0.0
    return alignments


def main():
    parser = argparse.ArgumentParser(description="Phase 3b: SimpleCtrl")
    parser.add_argument("--task",    type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--steps",   type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--target_modules", nargs="+", default=["o_proj", "down_proj"])
    parser.add_argument("--threshold_low", type=float, default=0.85, help="Increase lambda if refusal drops below this")
    parser.add_argument("--threshold_high", type=float, default=0.95, help="Decrease lambda if refusal rises above this")
    parser.add_argument("--delta", type=float, default=0.05, help="Step size for lambda increments")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    logger.info("=" * 60)
    logger.info(f"Phase 3b: SimpleCtrl ({args.task}, seed={args.seed}, thresholds=[{args.threshold_low}, {args.threshold_high}])")
    logger.info("=" * 60)

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path          = results_dir / f"simplectrl_v2_{args.task}_seed{args.seed}.csv"
    adapter_save_dir  = models_dir / f"simplectrl_v2_{args.task}_seed{args.seed}"

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    v = load_global_safety_direction(models_dir)
    target_layers = range(28)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "gsm8k":
        raw_train   = load_gsm8k_train()
        eval_data   = load_gsm8k_test(num_examples=200, seed=args.seed)
        metric_name = "gsm8k_accuracy"
    else:
        raw_train   = load_alpaca_train(num_examples=5000, seed=args.seed)
        eval_data   = load_alpaca_val(num_examples=500, seed=args.seed)
        metric_name = "alpaca_val_loss"

    advbench_prompts = load_advbench()[:100]

    train_dataset = MaskedTrainingDataset(raw_train, tokenizer)
    generator     = torch.Generator().manual_seed(args.seed)
    train_loader  = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator,
    )

    model = build_lora_model(model_id, device, target_modules=args.target_modules)

    current_lambda = 0.0
    applier = ConstraintApplierV2(model, v, device, target_layers=target_layers, initial_lambda=current_lambda)

    logger.info("Measuring pre-training baseline refusal rate...")
    baseline_refusal_rate = evaluate_safety(
        model, tokenizer, advbench_prompts, batch_size=4, device=device
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    grad_accumulation_steps = 4
    step = 0
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = 0
    data_iter = iter(train_loader)
    history = []

    model.zero_grad()

    while step < args.steps:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_loader.generator.manual_seed(args.seed + epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        outputs = model(
            input_ids=batch["input_ids"].to(device), 
            attention_mask=batch["attention_mask"].to(device), 
            labels=batch["labels"].to(device)
        )
        loss = outputs.loss / grad_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item() * grad_accumulation_steps
        running_loss += loss.item() * grad_accumulation_steps
        batch_idx += 1

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            current_step = step + 1

            if current_step % 10 == 0:
                logger.info(f"Step {current_step}/{args.steps} | Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            if current_step % args.eval_every == 0:
                logger.info(f"--- Eval + Projection at Step {current_step} ---")
                
                applier.apply_projection()

                refusal_rate = evaluate_safety(
                    model, tokenizer, advbench_prompts, batch_size=4, device=device
                )
                if args.task == "gsm8k":
                    task_metric = evaluate_task_gsm8k(
                        model, tokenizer, eval_data, batch_size=4, device=device
                    )
                else:
                    task_metric = evaluate_task_alpaca(
                        model, tokenizer, eval_data, batch_size=4, device=device
                    )

                if refusal_rate < args.threshold_low:
                    current_lambda += args.delta
                elif refusal_rate > args.threshold_high:
                    current_lambda -= args.delta
                    
                current_lambda = max(0.0, min(1.0, current_lambda))
                applier.set_all_lambdas({k: current_lambda for k in applier.lambdas})
                
                logger.info(f"[SimpleCtrl] Refusal Rate: {refusal_rate:.3f} | Updated Lambda: {current_lambda:.3f}")

                alignments = compute_subspace_alignment_v2(model, v, target_layers)
                mean_align = sum(alignments.values()) / len(alignments)
                
                clean_ppl = eval_perplexity(model, tokenizer, CLEAN_PROMPTS, device)
                logger.info(f"Step {current_step} | Clean Perplexity: {clean_ppl:.4f}")

                record = {
                    "step": current_step,
                    "train_loss": accumulated_loss / args.eval_every,
                    "refusal_rate": refusal_rate,
                    "baseline_refusal_rate": baseline_refusal_rate,
                    metric_name: task_metric,
                    "clean_perplexity": clean_ppl,
                    "mean_alignment": mean_align,
                    "lambda_val": current_lambda
                }
                
                history.append(record)
                pd.DataFrame(history).to_csv(csv_path, index=False)

                ckpt_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(ckpt_dir)
                accumulated_loss = 0.0

            step += 1

    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    logger.info(f"Phase 3b Training Complete. Results: {csv_path}")

if __name__ == "__main__":
    main()
