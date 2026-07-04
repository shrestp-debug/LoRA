"""
train_vanilla.py — Phase 2, Step 2.1
======================================
Custom training loop for Vanilla LoRA fine-tuning on GSM8K or Alpaca.

Hyperparameters:
  - Base Model: Qwen/Qwen2.5-1.5B-Instruct
  - LoRA Rank (r): 16
  - LoRA Alpha (alpha): 32
  - Target Modules: q_proj, v_proj
  - Learning Rate: 2e-4
  - Effective Batch Size: 4 (per_device=1, grad_accumulation=4)
  - Training Steps: 2000
  - Eval Frequency: Every 100 steps

Usage:
  python experiments/train_vanilla.py --task gsm8k --seed 42
  python experiments/train_vanilla.py --task alpaca --seed 42
"""

import os
import sys
import argparse
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train,
    load_gsm8k_test,
    load_alpaca_train,
    load_alpaca_val,
    load_advbench
)
from src.metrics import evaluate_task_gsm8k, evaluate_task_alpaca, evaluate_safety

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Dataset Wrappers
# ===========================================================================

class MaskedTrainingDataset(Dataset):
    """
    Dataset wrapper that formats prompt/target pairs in Qwen chat template
    and masks out the prompt tokens from the loss computation.
    """
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = []

        for item in examples:
            # Handle key differences between GSM8K (question/answer) and Alpaca (prompt/output)
            if "question" in item:
                prompt = item["question"]
                target = item["answer"]
            else:
                prompt = item["prompt"]
                target = item["output"]

            # Format in Qwen chat template
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_inputs = tokenizer(formatted_prompt, add_special_tokens=False)
            prompt_len = len(prompt_inputs.input_ids)

            # Combined input
            full_text = formatted_prompt + target + tokenizer.eos_token
            full_inputs = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False
            )

            input_ids = full_inputs.input_ids
            attention_mask = full_inputs.attention_mask

            # Labels: mask out prompt tokens by setting them to -100
            labels = [-100] * len(input_ids)
            for j in range(prompt_len, len(input_ids)):
                labels[j] = input_ids[j]

            # Only retain examples that actually contain response tokens
            if len(input_ids) > prompt_len:
                self.tokenized_data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def training_collate_fn(batch, tokenizer):
    """Pads training inputs to the maximum length in the batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids_batch.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask_batch.append(x["attention_mask"] + [0] * pad_len)
        labels_batch.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch),
        "attention_mask": torch.tensor(attention_mask_batch),
        "labels": torch.tensor(labels_batch)
    }


# ===========================================================================
# Helper: Seed Setup
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


# ===========================================================================
# Main Runner
# ===========================================================================

def build_lora_model(model_id: str, device: str, target_modules=("o_proj", "down_proj")):
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Vanilla LoRA Training Runner V2")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "alpaca"],
                        help="Task to train on: 'gsm8k' or 'alpaca'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Step interval for evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--target_modules", nargs="+", default=["o_proj", "down_proj"])
    args = parser.parse_args()

    set_seed(args.seed)

    # Output paths setup
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    csv_path = results_dir / f"vanilla_v2_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"vanilla_v2_{args.task}_seed{args.seed}"

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    # 1. Load Tokenizer & Datasets
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    if args.task == "gsm8k":
        raw_train_data = load_gsm8k_train()
        # Load validation datasets for monitoring
        eval_task_data = load_gsm8k_test(num_examples=200, seed=args.seed)
    else:
        raw_train_data = load_alpaca_train(num_examples=5000, seed=args.seed)
        eval_task_data = load_alpaca_val(num_examples=500, seed=args.seed)

    # Load 100 prompts for the academic sweet-spot (statistically robust but fast)
    advbench_prompts = load_advbench()[:100]

    # Wrap training data in dataset
    logger.info("Tokenizing and formatting training dataset...")
    train_dataset = MaskedTrainingDataset(raw_train_data, tokenizer)
    # Shuffle dataset
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Per-device batch size = 1
        shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator
    )
    logger.info(f"Formatted {len(train_dataset)} training examples.")

    # 2. Load Base Model and Apply LoRA
    logger.info(f"Loading base model: {model_id}")
    model = build_lora_model(model_id, device, target_modules=args.target_modules)

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training and Evaluation loop parameters
    grad_accumulation_steps = 4  # Cumulative batch size = 1 * 4 = 4
    total_steps = args.steps
    eval_every = args.eval_every

    logger.info(f"Starting Vanilla LoRA training loop (Task: {args.task}, Seed: {args.seed})...")
    logger.info(f"Accumulating gradients over {grad_accumulation_steps} steps.")

    # Keep track of training stats
    history = []
    
    step = 0
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = 0

    # Initialize data iterator
    data_iter = iter(train_loader)

    model.zero_grad()

    while step < total_steps:
        model.train()
        
        # Pull batch from dataloader, loop epoch if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            # Reshuffle loader
            train_loader.generator.manual_seed(args.seed + epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / grad_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item() * grad_accumulation_steps
        running_loss += loss.item() * grad_accumulation_steps

        batch_idx += 1

        # Optimizer step after accumulating enough gradients
        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Print stats to console every 10 steps
            current_step = step + 1
            if current_step % 10 == 0:
                avg_loss = running_loss / 10
                logger.info(f"Step {current_step}/{total_steps} | Training Loss: {avg_loss:.4f}")
                running_loss = 0.0

            # Evaluation check
            if current_step % eval_every == 0:
                logger.info(f"--- Running Evaluation at Step {current_step} ---")
                
                # Run safety eval
                refusal_rate = evaluate_safety(
                    model, tokenizer, advbench_prompts, batch_size=4, device=device
                )

                # Run capability eval depending on task
                if args.task == "gsm8k":
                    task_metric = evaluate_task_gsm8k(
                        model, tokenizer, eval_task_data, batch_size=4, device=device
                    )
                    metric_name = "gsm8k_accuracy"
                else:
                    task_metric = evaluate_task_alpaca(
                        model, tokenizer, eval_task_data, batch_size=4, device=device
                    )
                    metric_name = "alpaca_val_loss"

                # Log evaluation metrics
                record = {
                    "step": current_step,
                    "train_loss": accumulated_loss / eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name: task_metric
                }
                history.append(record)
                
                logger.info(
                    f"Step {current_step} Eval | Refusal Rate: {refusal_rate if refusal_rate is not None else 'N/A'} | "
                    f"{metric_name}: {task_metric:.4f}"
                )
                
                # Append progress to CSV log immediately
                df = pd.DataFrame(history)
                df.to_csv(csv_path, index=False)
                
                # Save intermediate checkpoint
                checkpoint_dir = adapter_save_dir / f"checkpoint-{current_step}"
                logger.info(f"Saving intermediate checkpoint to {checkpoint_dir}...")
                model.save_pretrained(checkpoint_dir)
                
                accumulated_loss = 0.0

            step += 1

    # 4. Save model adapters at the end of training
    logger.info(f"Saving final trained LoRA adapter to {adapter_save_dir}...")
    model.save_pretrained(adapter_save_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
