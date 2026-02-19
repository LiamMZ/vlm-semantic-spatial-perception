#!/usr/bin/env python3
"""
Test the GenAI interface with full response introspection and logprobs.

Sends prompts through the google.genai client and dumps the entire response
object — including candidates, logprobs, finish reason, safety ratings, and
usage metadata — so you can evaluate output shapes and token-level log
probabilities.

Usage:
    export GEMINI_API_KEY="your-key"
    python -m examples.test_genai_interface
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import is_dataclass, asdict
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from pprint import pformat
from textwrap import indent

from google import genai
from google.genai import types

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.genai_logging import configure_genai_logging
load_dotenv(PROJECT_ROOT / ".env")  # Load environment variables from .env if present
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-2.0-flash"
TOP_LOGPROBS = 5  # number of top logprobs per token (max 20)
LOG_DIR = PROJECT_ROOT / "outputs" / "genai_test_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")

# Test prompts: (label, prompt_text, config_overrides)
TEST_PROMPTS = [
    # 1. Simple factual question
    (
        "simple_factual",
        "What are the three laws of robotics as stated by Isaac Asimov? List them concisely.",
        {"temperature": 0.2, "max_output_tokens": 512},
    ),
    # 2. Structured JSON output
    (
        "structured_json",
        (
            "List 3 common household objects a robot arm could pick up. "
            "For each, provide: name, approximate weight in grams, and a "
            "suggested grasp type (pinch or power). Respond in JSON only."
        ),
        {
            "temperature": 0.3,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        },
    ),
    # 3. Robotics spatial reasoning
    (
        "spatial_reasoning",
        (
            "A red mug is at position [0.3, 0.5, 0.0] meters on a table. "
            "A blue plate is at [0.6, 0.5, 0.0]. Describe the spatial "
            "relationship between them and suggest a pick-and-place sequence "
            "to put the mug on the plate."
        ),
        {"temperature": 0.4, "max_output_tokens": 1024},
    ),
    # 4. PDDL generation (mirrors LLMTaskAnalyzer usage)
    (
        "pddl_generation",
        (
            "Generate a minimal PDDL domain for a pick-and-place robot. "
            "Include types (object, surface), predicates (on, holding, hand-empty), "
            "and actions (pick, place). Respond with valid PDDL only."
        ),
        {"temperature": 0.1, "max_output_tokens": 2048},
    ),
    # 5. Creative / high-temperature
    (
        "creative_high_temp",
        (
            "Imagine you are a robot arm in a kitchen. Describe in first person "
            "what you see and how you would make a cup of coffee, step by step."
        ),
        {"temperature": 0.9, "max_output_tokens": 1024},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(char: str = "=", width: int = 80) -> str:
    return char * width


def _to_dict(obj: object) -> object:
    """Recursively convert SDK objects to plain dicts for printing/serialization."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if is_dataclass(obj):
        return _to_dict(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return _to_dict(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


def _dump_logprobs(candidate: object) -> dict | None:
    """Extract and summarize logprobs from a candidate, returning shape info."""
    logprobs_result = getattr(candidate, "logprobs_result", None)
    if logprobs_result is None:
        return None

    chosen = getattr(logprobs_result, "chosen_candidates", None) or []
    top = getattr(logprobs_result, "top_candidates", None) or []

    summary = {
        "num_chosen_tokens": len(chosen),
        "num_top_candidate_steps": len(top),
    }

    # Shape of top_candidates: list of steps, each step has a list of candidates
    if top:
        step_sizes = [len(getattr(step, "candidates", []) or []) for step in top]
        summary["top_candidates_per_step"] = {
            "min": min(step_sizes),
            "max": max(step_sizes),
            "shape": f"[{len(top)} steps x {step_sizes[0] if len(set(step_sizes)) == 1 else 'variable'} candidates]",
        }

    # Print first few chosen tokens with their logprobs
    chosen_sample = []
    for entry in chosen[:10]:
        token = getattr(entry, "token", None)
        log_probability = getattr(entry, "log_probability", None)
        chosen_sample.append({"token": token, "log_probability": log_probability})
    summary["chosen_tokens_sample (first 10)"] = chosen_sample

    # Print first few top-k steps
    top_sample = []
    for step in top[:5]:
        candidates = getattr(step, "candidates", []) or []
        step_entries = []
        for c in candidates:
            step_entries.append({
                "token": getattr(c, "token", None),
                "log_probability": getattr(c, "log_probability", None),
            })
        top_sample.append(step_entries)
    summary["top_candidates_sample (first 5 steps)"] = top_sample

    return summary


def run_prompt(
    client: genai.Client,
    label: str,
    prompt: str,
    config_overrides: dict,
    index: int,
) -> dict:
    """Send a single prompt, dump the full response object, and return results."""
    print(f"\n{_separator()}")
    print(f"  [{index}] {label}")
    print(_separator())
    print(f"  Prompt ({len(prompt)} chars):")
    print(indent(prompt, "    "))

    # Merge per-prompt overrides with logprobs settings
    merged_config = {
        "response_logprobs": True,
        "logprobs": TOP_LOGPROBS,
        **config_overrides,
    }
    print(f"  Config: {merged_config}")
    print(_separator("-"))

    config = types.GenerateContentConfig(**merged_config)

    start = time.time()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=config,
    )
    elapsed = time.time() - start

    # --- Full response object dump ---
    print(f"\n  Full response object (type: {type(response).__name__}):")
    print(_separator("-"))
    response_dict = _to_dict(response)
    print(indent(pformat(response_dict, width=120, depth=6), "    "))
    print(_separator("-"))

    # --- Response text ---
    text = getattr(response, "text", None)
    if text is None:
        text = "<no text in response>"
    print(f"\n  Response text ({elapsed:.2f}s):")
    print(indent(text, "    "))

    # --- Candidates ---
    candidates = getattr(response, "candidates", []) or []
    print(f"\n  Candidates: {len(candidates)}")
    for ci, cand in enumerate(candidates):
        print(f"    candidate[{ci}]:")
        print(f"      finish_reason : {getattr(cand, 'finish_reason', None)}")
        print(f"      safety_ratings: {_to_dict(getattr(cand, 'safety_ratings', None))}")
        print(f"      avg_logprobs  : {getattr(cand, 'avg_logprobs', None)}")

        # --- Logprobs ---
        lp_summary = _dump_logprobs(cand)
        if lp_summary:
            print(f"      logprobs_result:")
            print(indent(pformat(lp_summary, width=110), "        "))
        else:
            print(f"      logprobs_result: None")

    # --- Usage metadata ---
    usage = getattr(response, "usage_metadata", None)
    usage_dict = {}
    if usage:
        usage_dict = _to_dict(usage)
        print(f"\n  Usage metadata:")
        print(indent(pformat(usage_dict, width=100), "    "))

    # --- Model version ---
    model_version = getattr(response, "model_version", None)
    if model_version:
        print(f"\n  Model version: {model_version}")

    # Try pretty-printing if JSON
    try:
        parsed = json.loads(text)
        print(f"\n  Parsed JSON:")
        print(indent(json.dumps(parsed, indent=2), "    "))
    except (json.JSONDecodeError, TypeError):
        pass

    print()

    return {
        "label": label,
        "prompt": prompt,
        "config": merged_config,
        "response_text": text,
        "elapsed_seconds": round(elapsed, 3),
        "usage": usage_dict,
        "num_candidates": len(candidates),
        "logprobs_shape": lp_summary if candidates else None,
        "full_response": response_dict,
    }


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    # Enable GenAI logging to disk
    log_path = configure_genai_logging(LOG_DIR)
    print(f"GenAI logging enabled → {log_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Running {len(TEST_PROMPTS)} test prompts...\n")

    client = genai.Client(api_key=api_key)
    results = []

    for i, (label, prompt, config_overrides) in enumerate(TEST_PROMPTS, start=1):
        try:
            result = run_prompt(client, label, prompt, config_overrides, i)
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR on [{i}] {label}: {exc}")
            results.append({"label": label, "error": str(exc)})

    # Write summary
    summary_path = LOG_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{_separator('=')}")
    print(f"  Done. {len(results)} prompts executed.")
    print(f"  Summary → {summary_path}")
    print(f"  Full logs → {LOG_DIR}")
    print(_separator("="))


if __name__ == "__main__":
    main()
