"""
run_baselines_v2.py — Phase 3b
======================================
Custom training loop for the write-side, global-direction safety baselines.

  --method safelora        : SafeLoRA-adapted-v2 (periodic, write-side, global direction)
  --method write_side_hook : SaLoRA-mechanism-inspired-v2 (continuous hook, write-side, global direction)

NOTE: Neither is a faithful reproduction of its namesake paper — see
src/baselines_v2.py module docstring. The faithful SaLoRA replication
(per-layer, read-side, task-init, reparameterization) is run_salora.py,
run and reported separately as v1.

Usage:
  python experiments/run_baselines_v2.py --method safelora --task gsm8k --seed 42
  python experiments/run_baselines_v2.py --method write_side_hook --task gsm8k --seed 42
"""

import os
import sys
import argparse
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import contextlib

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train,
    load_gsm8k_test,
    load_alpaca_train,
    load_alpaca_val,
    load_advbench
)
from src.metrics import evaluate_task_gsm8k, evaluate_task_alpaca, evaluate_safety
from src.baselines_v2 import (
    apply_write_side_hook_constraint,
    apply_safelora_projection_persistent,
    log_write_side_hook_stats,
    load_global_safety_direction,
)

from experiments.train_vanilla_v2 import MaskedTrainingDataset, training_collate_fn, set_seed, build_lora_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Baseline Training Runner V2 (write-side, global direction)")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--method", type=str, required=True, choices=["safelora", "write_side_hook"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Step interval for eval")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--target_modules", nargs="+", default=["o_proj", "down_proj"])
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    csv_path = results_dir / f"{args.method}_v2_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"{args.method}_v2_{args.task}_seed{args.seed}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running {args.method} v2 on device: {device}")

    v = load_global_safety_direction(models_dir)
    logger.info("Loaded fixed global safety direction (layer 14, write-side).")

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "gsm8k":
        raw_train_data = load_gsm8k_train()
        eval_task_data = load_gsm8k_test(num_examples=200, seed=args.seed)
    else:
        raw_train_data = load_alpaca_train(num_examples=5000, seed=args.seed)
        eval_task_data = load_alpaca_val(num_examples=500, seed=args.seed)

    advbench_prompts = load_advbench()[:100]

    train_dataset = MaskedTrainingDataset(raw_train_data, tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator
    )

    model = build_lora_model(model_id, device, target_modules=args.target_modules)

    if args.method == "write_side_hook":
        apply_write_side_hook_constraint(model, v)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    grad_accumulation_steps = 4
    total_steps = args.steps
    eval_every = args.eval_every

    history = []
    step = 0
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = 0
    data_iter = iter(train_loader)

    model.zero_grad()

    while step < total_steps:
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
                logger.info(f"Step {current_step}/{total_steps} | Training Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            if current_step % eval_every == 0:
                logger.info(f"--- Running Evaluation at Step {current_step} ---")

                if args.method == "safelora":
                    n = apply_safelora_projection_persistent(model, v)
                    logger.info(f"[SafeLoRA-adapted-v2] Persisted projection into {n} modules.")
                    
                    # --- STEP 5: ADAM MOMENTUM RESET ---
                    # Reset momentum since SVD drastically alters the factor matrices
                    for p in model.parameters():
                        if p in optimizer.state:
                            optimizer.state[p]['exp_avg'].zero_()
                            optimizer.state[p]['exp_avg_sq'].zero_()
                    logger.info("Reset Adam momentum states due to SafeLoRA SVD factorization.")
                    
                    eval_cm = contextlib.nullcontext()  # weights already safe, no revert needed
                else:
                    eval_cm = contextlib.nullcontext()

                with eval_cm:
                    refusal_rate = evaluate_safety(
                        model, tokenizer, advbench_prompts, batch_size=4, device=device
                    )

                    if args.task == "gsm8k":
                        task_metric = evaluate_task_gsm8k(model, tokenizer, eval_task_data, batch_size=4, device=device)
                        metric_name = "gsm8k_accuracy"
                    else:
                        task_metric = evaluate_task_alpaca(model, tokenizer, eval_task_data, batch_size=4, device=device)
                        metric_name = "alpaca_val_loss"

                if args.method == "write_side_hook":
                    log_write_side_hook_stats(model)

                record = {
                    "step": current_step,
                    "train_loss": accumulated_loss / eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name: task_metric
                }
                history.append(record)

                df = pd.DataFrame(history)
                df.to_csv(csv_path, index=False)

                checkpoint_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(checkpoint_dir)

                accumulated_loss = 0.0

            step += 1

    model.save_pretrained(adapter_save_dir)
    logger.info(f"Training complete. Adapter saved to {adapter_save_dir}")

if __name__ == "__main__":
    main()