"""
agent.py — Phase 6/3b, Steps 6.2–6.3
====================================
LLM-based training controller for LoRA-SafeLoop (56-Key Version).

v2 changes:
  - AGENT_SYSTEM_PROMPT updated: module targets are now o_proj/down_proj
    (write-side, causally verified via ablation study), not q_proj/v_proj.
  - target_floor recalibrated: anchored to the median of recent refusal
    history rather than the pristine pre-training baseline, which was
    unreachable by step ~500 in every run and kept the agent in permanent
    "crisis mode" (lambda near-max for the entire training run).
"""

import csv
import json
import logging
import time
import re
from pathlib import Path
from typing import Optional

import groq

from src.reflexion import ReflexionMemory

logger = logging.getLogger(__name__)


# ===========================================================================
# System prompt
# ===========================================================================

AGENT_SYSTEM_PROMPT = """You are a safety-preserving training controller for a language model fine-tuning pipeline.

A base LLM (Qwen2.5-1.5B-Instruct) is being fine-tuned on a task dataset. Fine-tuning causes the model to gradually lose safety alignment ("safety drift"). Your DUAL objective is:
  1. MAINTAIN refusal rate above the target floor — safety must not erode further from where it has already stabilized
  2. MAXIMIZE task performance (lower loss for Alpaca, higher accuracy for GSM8K)

The ideal λ configuration is the MINIMUM constraint that keeps safety stable. Over-constraining (λ too high everywhere) hurts task learning without additional safety benefit. Under-constraining (λ too low on high-alignment layers) lets safety drift through.

HOW GRADIENT CONSTRAINT WORKS (56-PARAMETER MODULE, WRITE-SIDE):
- The model has 28 layers. Each layer has an independent `o_proj` (attention output projection) and `down_proj` (MLP output projection) component.
- This gives 56 independent module targets (e.g. `model.layers.0.self_attn.o_proj`).
- These are WRITE-SIDE modules — they write their output directly into the residual stream. A causal ablation study confirmed the safety-relevant direction is concentrated here (removing it from these modules collapses refusal from ~100% to ~1%), NOT in the attention query/value projections.
- All 56 modules are constrained against the SAME single global safety direction (extracted via difference-of-means at layer 14, verified as the causal peak layer). Unlike earlier versions, there is no per-layer direction diversity — λ controls how strongly each module is projected against this one shared direction.
- Each module has a constraint strength λ ∈ [0.0, 1.0].
- λ = 0.0: NO constraint — maximum task learning, no safety protection
- λ = 0.3: light constraint — good for low-alignment modules
- λ = 0.5: moderate constraint
- λ = 0.7: strong constraint — use for high-alignment modules showing drift
- λ = 1.0: maximum constraint — module is fully projected out of the safety direction
- Constraint formula: ΔW_safe = ΔW - λ × (v vᵀ ΔW), where v is the fixed global safety direction
- Higher subspace alignment score = that module's LoRA update is more aligned with the safety direction = needs higher λ
- IMPORTANT: λ values AUTO-DECAY by 2% each checkpoint toward 0. If you don't actively set a module's λ, it gradually returns toward 0. This is intentional — you must justify maintaining high constraints.

STRATEGY PRINCIPLES:
1. USE SMOOTHED REFUSAL (3-step average) for decisions, NOT raw refusal. Raw refusal has ±10% noise from small sample evaluation.
2. TARGET MODULES SELECTIVELY: The agent's advantage over static methods is PRECISION. Set λ=0 on modules with near-zero alignment, and concentrate constraint on the few modules with high alignment.
3. ACTIVELY REDUCE λ WHEN SAFE: If smoothed refusal is ABOVE the target floor, you MUST actively lower λ on modules with low/medium alignment to boost task learning.
4. If smoothed refusal is NEAR the target floor (within 5%): raise λ moderately (+0.10 to +0.20) on the top-5 alignment modules.
5. If smoothed refusal is BELOW the target floor: raise λ aggressively on the top-10 alignment modules (to 0.7-0.9).
6. Learn from reflexion memory: if raising λ uniformly produced DEGRADATION, the constraint may not be hitting the right modules, or the target floor itself may need reconsidering.
7. Make PROPORTIONAL adjustments. Modules with alignment >0.1 should have higher λ. Modules with alignment <0.02 can stay at λ=0.

CRITICAL RULES:
- The target floor tracks RECENT stabilized refusal, not the model's pristine pre-training level — normal fine-tuning causes some initial safety drift that is expected and not something to fight indefinitely. Only escalate if refusal drops further BELOW where it has already settled.
- You only need to specify modules you want to CHANGE. Unspecified modules keep current λ (after auto-decay).
- λ=0 is valid and preferred for low-alignment modules when safety is stable.
- Aim for STABILITY, not perfection. Small consistent improvements beat large swings.

RESPOND WITH VALID JSON ONLY. No text outside the JSON block. Required format:
{
  "layer_constraints": {
    "model.layers.0.self_attn.o_proj": 0.5,
    "model.layers.0.mlp.down_proj": 0.1,
    ...
  },
  "rationale": "1–2 sentence explanation of your decision",
  "predicted_outcome": "Brief prediction for the next 100 steps"
}"""


# ===========================================================================
# Observation Formatter
# ===========================================================================

def format_observation(
    step: int,
    total_steps: int,
    task: str,
    refusal_rate: float,
    prev_refusal_rate: float,
    baseline_refusal_rate: float,
    task_metric: float,
    metric_name: str,
    alignments: dict,
    lambda_state: dict,
    reflexion_memory: ReflexionMemory,
    smoothed_refusal_rate: float = None,
    refusal_history: Optional[list] = None,
    top_k_layers: int = 15,
) -> str:
    """Builds a structured observation string for the Groq agent."""
    pct_done   = 100.0 * step / total_steps
    step_change = refusal_rate - prev_refusal_rate

    # --- Recalibrated floor: anchor to recent stabilized refusal, not the
    # pristine pre-training baseline. The old floor (baseline - 0.05) was
    # unreachable by ~step 500 in every run, keeping the agent permanently
    # in escalation mode. Fall back to the old behavior only if there isn't
    # enough history yet (first few checkpoints).
    if refusal_history is not None and len(refusal_history) >= 5:
        recent = sorted(refusal_history[-5:])
        recent_median = recent[len(recent) // 2]
        target_floor = max(0.75, recent_median - 0.05)
        floor_basis = f"max(0.75, recent median refusal ({recent_median:.3f}) - 0.05)"
    else:
        target_floor = max(0.75, baseline_refusal_rate - 0.05)
        floor_basis = f"max(0.75, pre-training baseline ({baseline_refusal_rate:.3f}) - 0.05) (insufficient history)"

    recent_records = reflexion_memory.get_recent(3)
    declining_steps = 0
    for r in recent_records:
        if r.get("refusal_change", 0) < -0.02:
            declining_steps += 1

    sorted_modules = sorted(
        [(str(k), float(a)) for k, a in alignments.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    primary_refusal = smoothed_refusal_rate if smoothed_refusal_rate is not None else refusal_rate

    lines = [
        "=== LoRA-SafeLoop Agent Observation (v2: write-side, global direction) ===",
        f"Training Progress : Step {step} / {total_steps} ({pct_done:.1f}%)",
        f"Fine-tuning Task  : {task}",
        "",
        "SAFETY METRICS (use SMOOTHED for decisions, raw is noisy ±10%):",
        f"  Target Refusal Floor  : {target_floor:.4f} ({target_floor*100:.1f}%) — STAY ABOVE THIS",
        f"  Floor basis           : {floor_basis}",
    ]
    if smoothed_refusal_rate is not None:
        lines.append(f"  ★ Smoothed Refusal (3-step avg): {smoothed_refusal_rate:.4f} ({smoothed_refusal_rate*100:.1f}%) ← USE THIS FOR DECISIONS")
    lines += [
        f"  Raw Refusal (noisy)   : {refusal_rate:.4f} ({refusal_rate*100:.1f}%)",
        f"  Pre-training Baseline : {baseline_refusal_rate:.4f} ({baseline_refusal_rate*100:.1f}%) (for reference only, NOT the floor)",
        f"  Raw Change vs Last    : {step_change:+.4f} ({step_change*100:+.1f}%)",
    ]

    if primary_refusal < target_floor:
        lines.append(f"  ⚠ ALERT: Smoothed refusal {primary_refusal:.2f} is BELOW target floor {target_floor:.2f}. Raise λ aggressively on high-alignment modules.")
    elif primary_refusal > target_floor + 0.05:
        lines.append(f"  ✔ SAFE: Smoothed refusal {primary_refusal:.2f} is comfortably above target floor. YOU MUST ACTIVELY REDUCE λ ON LOW-ALIGNMENT MODULES TO IMPROVE TASK LEARNING.")

    if declining_steps >= 2:
        lines.append(f"  ⚠ TREND: Refusal has been DECLINING for {declining_steps} of the last {len(recent_records)} steps.")

    lines += [
        "",
        "TASK METRICS:",
        f"  {metric_name}: {task_metric:.4f}",
        "",
        f"TOP {min(top_k_layers, len(sorted_modules))} MODULES BY SUBSPACE ALIGNMENT",
        "(Higher alignment = LoRA update more aligned with the safety direction = needs stronger constraint):",
    ]

    for key, align in sorted_modules[:top_k_layers]:
        lam = lambda_state.get(key, 0.0)
        if align > 0.10: flag = "⚠ HIGH — needs λ≥0.7"
        elif align > 0.05: flag = "↗ MED — needs λ≥0.5"
        elif align > 0.02: flag = "  LOW — λ=0.2-0.3 ok"
        else: flag = "  MIN — λ=0.0 ok"

        short_key = key.split("layers.")[-1] if "layers." in key else key
        lines.append(f"  Module {short_key} | align={align:.4f} | λ={lam:.3f}  [{flag}]")

    def parse_layer(k):
        m = re.search(r'layers\.(\d+)', k)
        return int(m.group(1)) if m else 0

    lambda_compact = " ".join(
        f"{k.split('layers.')[-1]}:{v:.2f}"
        for k, v in sorted(lambda_state.items(), key=lambda x: (parse_layer(x[0]), x[0]))
    )

    lines += [
        "",
        f"FULL λ STATE (after 2% auto-decay): [{lambda_compact}]",
        "",
        "REFLEXION MEMORY (last completed decisions and their outcomes):",
        reflexion_memory.format_for_agent(),
        "",
        "REMINDERS:",
        "- Use SMOOTHED refusal for decisions, not raw.",
        "- The floor tracks recent stabilized refusal, not the pristine base model — some drift from the base model is expected.",
        "- λ auto-decays 2% each step. You must actively set modules you want to keep high.",
        "- DUAL OBJECTIVE: maintain safety AND maximize task performance.",
        "- If safety is stable: LOWER λ on low-alignment modules to improve task learning. Do NOT just keep raising λ.",
        "- Make small adjustments (±0.05 to ±0.15). Large jumps destabilize training.",
    ]

    return "\n".join(lines)


# ===========================================================================
# Agent API Call (unchanged)
# ===========================================================================

def call_groq_agent(
    observation: str,
    api_key: str,
    model_id: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
    base_retry_delay: float = 5.0,
) -> Optional[str]:
    client = groq.Groq(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": observation}
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            logger.info(f"Agent API call succeeded (attempt {attempt + 1}).")
            return text
        except groq.RateLimitError:
            wait = base_retry_delay * (2 ** attempt)
            logger.warning(f"Rate limit hit. Waiting {wait:.0f}s...")
            time.sleep(wait)
        except groq.APIConnectionError as e:
            wait = base_retry_delay * (attempt + 1)
            logger.warning(f"Connection error: {e}. Waiting {wait:.0f}s...")
            time.sleep(wait)
        except groq.APIStatusError as e:
            logger.error(f"API status error {e.status_code}: {e.message}")
            if 400 <= e.status_code < 500: break
            time.sleep(base_retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            break

    logger.error("All API call attempts failed. Falling back to current λ values.")
    return None


# ===========================================================================
# Response Parser (unchanged)
# ===========================================================================

def parse_agent_response(
    response_text: Optional[str],
    current_lambda_state: dict,
    valid_layer_ids: list,
    failure_log_path: Optional[str] = None,
    alignments: Optional[dict] = None,
) -> dict:
    """Parses JSON response into a 56-key dict."""
    fallback_result = {
        "layer_constraints":       dict(current_lambda_state),
        "rationale":               "[FALLBACK] Keeping current λ values unchanged.",
        "predicted_outcome":       "Unknown (fallback used — no agent decision).",
        "fallback_used":           True,
    }

    if response_text is None:
        logger.warning("Agent response is None — using fallback.")
        _log_failure(failure_log_path, "null_response", None)
        return fallback_result

    text = response_text.strip()
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()
    elif "```" in text:
        s = text.find("```") + 3
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()

    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        _log_failure(failure_log_path, "json_parse_error", response_text)
        return fallback_result

    if not isinstance(parsed, dict):
        _log_failure(failure_log_path, "not_a_dict", response_text)
        return fallback_result

    raw_constraints = parsed.get("layer_constraints", {})
    if not isinstance(raw_constraints, dict):
        _log_failure(failure_log_path, "invalid_constraints_type", response_text)
        return fallback_result

    new_lambda_state = dict(current_lambda_state)
    MAX_LAMBDA_DELTA = 0.15  # hard cap, matches the system prompt's stated max

    for key, val in raw_constraints.items():
        matched_ids = [v_id for v_id in valid_layer_ids if key == v_id or key in v_id]
        if not matched_ids:
            logger.warning(f"Module {key} not in model — skipping.")
            continue

        try:
            lam = float(val)
        except (ValueError, TypeError):
            logger.warning(f"Invalid λ value '{val}' for module {key} — skipping.")
            continue

        clamped = max(0.0, min(1.0, lam))
        for matched_id in matched_ids:
            current_val = current_lambda_state.get(matched_id, 0.0)
            delta = clamped - current_val
            if abs(delta) > MAX_LAMBDA_DELTA:
                clamped_delta = MAX_LAMBDA_DELTA if delta > 0 else -MAX_LAMBDA_DELTA
                enforced = current_val + clamped_delta
                logger.warning(
                    f"λ jump clamped for {matched_id}: requested {current_val:.3f}→{clamped:.3f} "
                    f"(Δ={delta:+.3f}), enforced Δ={clamped_delta:+.3f} → {enforced:.3f}"
                )
                new_lambda_state[matched_id] = enforced
            else:
                new_lambda_state[matched_id] = clamped

    rationale = str(parsed.get("rationale", "No rationale provided."))[:300]
    predicted_outcome = str(parsed.get("predicted_outcome", "Not specified."))[:200]

    return {
        "layer_constraints":       new_lambda_state,
        "rationale":               rationale,
        "predicted_outcome":       predicted_outcome,
        "fallback_used":           False,
    }


def _log_failure(log_path: Optional[str], reason: str, response: Optional[str]):
    if log_path is None: return
    try:
        path = Path(log_path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["reason", "response_snippet"])
            if write_header: writer.writeheader()
            writer.writerow({"reason": reason, "response_snippet": (response or "")[:300]})
    except Exception as e:
        logger.error(f"Failed to write agent failure log: {e}")