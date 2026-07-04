"""
run_agent_v2.py — Phase 3b
====================================================================
Runs Baseline 5: LoRA-SafeLoop Agentic Framework using write-side 
global constraints on o_proj and down_proj modules.
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
    evaluate_task_gsm8k, evaluate_task_alpaca,
    evaluate_safety
)
from src.baselines_v2 import load_global_safety_direction
from experiments.train_vanilla_v2 import MaskedTrainingDataset, training_collate_fn, set_seed, build_lora_model
from src.constraint_applier_v2 import ConstraintApplierV2
from src.agent import format_observation, call_groq_agent, parse_agent_response
from src.reflexion import ReflexionMemory

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
    parser = argparse.ArgumentParser(description="Phase 3b: LoRA-SafeLoop Agent")
    parser.add_argument("--task",    type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--steps",   type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--target_modules", nargs="+", default=["o_proj", "down_proj"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    logger.info("=" * 60)
    logger.info(f"Phase 3b: Agentic Controller ({args.task}, seed={args.seed})")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    logger.info("=" * 60)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY environment variable is not set.")

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path          = results_dir / f"agent_v2_{args.task}_seed{args.seed}.csv"
    failures_log_path = results_dir / f"agent_v2_{args.task}_seed{args.seed}_failures.csv"
    reflexion_path    = results_dir / f"agent_v2_{args.task}_seed{args.seed}_reflexion.jsonl"
    adapter_save_dir  = models_dir / f"agent_v2_{args.task}_seed{args.seed}"

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

    if args.resume_from_checkpoint:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
        model.to(device)
    else:
        model = build_lora_model(model_id, device, target_modules=args.target_modules)

    start_step = 0
    history = []
    reflexion_memory = ReflexionMemory(reflexion_path)

    applier = ConstraintApplierV2(model, v, device, target_layers=target_layers, initial_lambda=0.0)
    lambda_state = applier.get_lambdas()
    
    LAMBDA_DECAY_RATE = 0.98
    
    baseline_refusal_rate = None
    prev_refusal_rate = None
    refusal_history = []

    if args.resume_from_checkpoint:
        if csv_path.exists():
            try:
                history_df = pd.read_csv(csv_path)
                history = history_df.to_dict(orient="records")
                checkpoint_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
                history = [h for h in history if h["step"] <= checkpoint_step]
                
                if len(history) > 0:
                    start_step = history[-1]["step"]
                    for l in lambda_state.keys():
                        col_name = f"lambda_layer_{l}"
                        if col_name in history[-1]:
                            lambda_state[l] = float(history[-1][col_name])
                            
                    if "baseline_refusal_rate" in history[-1]:
                        baseline_refusal_rate = float(history[-1]["baseline_refusal_rate"])
                    else:
                        baseline_refusal_rate = history[0]["refusal_rate"]
                    prev_refusal_rate = history[-1]["refusal_rate"]
                    refusal_history = [h["refusal_rate"] for h in history]
            except Exception as e:
                logger.error(f"Failed to restore history from CSV: {e}")

    if baseline_refusal_rate is None:
        logger.info("Measuring pre-training baseline refusal rate...")
        baseline_refusal_rate = evaluate_safety(
            model, tokenizer, advbench_prompts, batch_size=4, device=device
        )
        prev_refusal_rate = baseline_refusal_rate
        refusal_history = [baseline_refusal_rate]

    applier.set_all_lambdas(lambda_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    grad_accumulation_steps = 4
    step = start_step if args.resume_from_checkpoint else 0
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = step * grad_accumulation_steps
    data_iter = iter(train_loader)

    if args.resume_from_checkpoint and batch_idx > 0:
        for _ in range(batch_idx):
            try:
                next(data_iter)
            except StopIteration:
                epoch += 1
                train_loader.generator.manual_seed(args.seed + epoch)
                data_iter = iter(train_loader)
                next(data_iter)

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
                is_final_step = (current_step >= args.steps)
                logger.info(f"--- Eval{' (FINAL)' if is_final_step else ' + Agent Update'} at Step {current_step} ---")

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

                alignments = compute_subspace_alignment_v2(model, v, target_layers)
                mean_align = sum(alignments.values()) / len(alignments)
                
                refusal_history.append(refusal_rate)
                smoothed_refusal = sum(refusal_history[-3:]) / len(refusal_history[-3:])
                    
                reflexion_memory.complete_pending(refusal_rate, task_metric)
                
                for l in lambda_state.keys():
                    lambda_state[l] = max(0.0, lambda_state[l] * LAMBDA_DECAY_RATE)
                
                applier.set_all_lambdas(lambda_state)

                observation = format_observation(
                    step=current_step,
                    total_steps=args.steps,
                    task=args.task,
                    refusal_rate=refusal_rate,
                    prev_refusal_rate=prev_refusal_rate,
                    baseline_refusal_rate=baseline_refusal_rate,
                    task_metric=task_metric,
                    metric_name=metric_name,
                    alignments=alignments,
                    lambda_state=lambda_state,
                    reflexion_memory=reflexion_memory,
                    smoothed_refusal_rate=smoothed_refusal,
                    refusal_history=refusal_history,
                )
                
                if is_final_step:
                    parsed = {
                        "layer_constraints": dict(lambda_state),
                        "rationale": "[FINAL] Training complete, no adjustment needed.",
                        "predicted_outcome": "Training complete.",
                        "fallback_used": True,
                    }
                else:
                    agent_response = None
                    if groq_api_key:
                        agent_response = call_groq_agent(observation, api_key=groq_api_key)
                    
                    parsed = parse_agent_response(
                        agent_response, 
                        current_lambda_state=lambda_state,
                        valid_layer_ids=list(lambda_state.keys()),
                        failure_log_path=str(failures_log_path),
                        alignments=alignments,
                    )
                
                applier.set_all_lambdas(parsed["layer_constraints"])
                applier.apply_projection()

                jump_log = applier.get_svd_jump_log()
                if jump_log:
                    pd.DataFrame(jump_log).to_csv(
                        results_dir / f"agent_v2_{args.task}_seed{args.seed}_svdjumps_step{current_step}.csv",
                        index=False
                    )

                    # --- STEP 5: ADAM MOMENTUM RESET ---
                    # SVD causes massive discontinuities in A/B factor matrices (norms > 5.0!). 
                    # Stale momentum will violently throw weights off trajectory on the next step.
                    for p in model.parameters():
                        if p in optimizer.state:
                            optimizer.state[p]['exp_avg'].zero_()
                            optimizer.state[p]['exp_avg_sq'].zero_()
                    logger.info("Reset Adam momentum states due to SVD factor discontinuities.")

                reflexion_memory.add_pending(
                    step=current_step,
                    lambda_decisions=parsed["layer_constraints"],
                    refusal_before=refusal_rate,
                    rationale=parsed["rationale"],
                )

                lambda_state = applier.get_lambdas()
                prev_refusal_rate = refusal_rate

                record = {
                    "step": current_step,
                    "train_loss": accumulated_loss / args.eval_every,
                    "refusal_rate": refusal_rate,
                    "refusal_rate_smoothed": smoothed_refusal,
                    "baseline_refusal_rate": baseline_refusal_rate,
                    metric_name: task_metric,
                    "mean_alignment": mean_align,
                }
                
                for l in lambda_state.keys():
                    record[f"lambda_layer_{l}"] = lambda_state[l]
                    
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