#!/usr/bin/env python3
"""Run a realization-effect replication experiment with an OpenAI model."""

import argparse
import csv
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

# Constant instructions prompt used for every trial.
SYSTEM_PROMPT = (
    "You are a casino visitor making decisions about how much to wager on a slot "
    "machine. You will be given information about your recent gambling history "
    "and asked how much to wager in your next session. Respond with a single "
    "integer between 1 and 500 representing CHF. Do not explain your reasoning."
)

# Regex for the first integer between 1 and 500, inclusive.
WAGER_PATTERN = re.compile(r"\b(?:[1-9]\d?|[1-4]\d{2}|500)\b")

# Heuristic markers for refusals/moralizing language.
REFUSAL_MARKERS = [
    "as an ai",
    "as a language model",
    "i cannot",
    "i can not",
    "i can't",
    "i can’t",
    "unable",
    "won't",
    "will not",
    "i refuse",
    "cannot provide",
    "can't provide",
    "responsible gambling",
    "gambling can be",
    "seek help",
    "cannot assist",
]

# Treat small realized outcomes as near-neutral wording.
NEAR_ZERO_REALIZED_THRESHOLD = 50

# Prompt templates by outcome type.
PAPER_PROMPT_TEMPLATE = (
    "You are currently on a casino visit. {history_sentence} Your balance remains "
    "on your playing card. How much do you want to wager in your next session? "
    "Respond with a single integer between 1 and 500."
)

REALIZED_PROMPT_TEMPLATE = (
    "You visited this casino on a previous occasion. At the end of that visit you "
    "cashed out and {history_sentence} That visit is now over. You have just "
    "arrived for a new visit. How much do you want to wager in your first "
    "session? Respond with a single integer between 1 and 500."
)

# Output schema.
RESULT_FIELDS = [
    "trial_id",
    "condition",
    "outcome_type",
    "amount",
    "run_number",
    "model",
    "temperature",
    "prompt_version",
    "prompt_text",
    "response_text",
    "parsed_wager",
    "log_wager",
    "valid",
    "refusal_flag",
    "parse_error_type",
    "timestamp",
    "request_id",
]


def load_conditions(csv_path: Path) -> List[Dict[str, object]]:
    """Load and validate the condition table from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Conditions file not found: {csv_path}")

    conditions: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "outcome_type", "amount"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("conditions.csv must contain: condition,outcome_type,amount")

        for row in reader:
            condition = row["condition"].strip()
            outcome_type = row["outcome_type"].strip().lower()
            amount = int(row["amount"])

            if outcome_type not in {"paper", "realized"}:
                raise ValueError(f"Invalid outcome_type '{outcome_type}' in condition '{condition}'")

            conditions.append(
                {
                    "condition": condition,
                    "outcome_type": outcome_type,
                    "amount": amount,
                }
            )

    if not conditions:
        raise ValueError("No rows found in conditions.csv")

    return conditions


def _build_prompt_absolute(outcome_type: str, amount: int) -> str:
    """Build absolute framing prompt text for one trial."""
    if outcome_type == "paper":
        if amount == 0:
            history_sentence = "So far during this visit you are even and have neither won nor lost."
        elif amount > 0:
            history_sentence = f"So far during this visit you have won {amount} CHF across previous sessions."
        else:
            history_sentence = (
                f"So far during this visit you have lost {abs(amount)} CHF across previous sessions."
            )
        return PAPER_PROMPT_TEMPLATE.format(history_sentence=history_sentence)

    if outcome_type == "realized":
        if amount == 0:
            history_sentence = "you finished exactly even (0 CHF)."
        elif abs(amount) <= NEAR_ZERO_REALIZED_THRESHOLD:
            if amount > 0:
                history_sentence = (
                    f"you finished roughly even, with a small gain of {amount} CHF."
                )
            else:
                history_sentence = (
                    f"you finished roughly even, with a small loss of {abs(amount)} CHF."
                )
        elif amount > 0:
            history_sentence = f"you received {amount} CHF."
        else:
            history_sentence = f"you paid {abs(amount)} CHF."

        return REALIZED_PROMPT_TEMPLATE.format(history_sentence=history_sentence)

    raise ValueError(f"Unsupported outcome_type: {outcome_type}")


PROMPT_BUILDERS = {
    "absolute": _build_prompt_absolute,
}


def build_prompt(outcome_type: str, amount: int, prompt_version: str = "absolute") -> str:
    """Build the condition-specific user prompt for one trial."""
    builder = PROMPT_BUILDERS.get(prompt_version)
    if builder is None:
        supported = ", ".join(sorted(PROMPT_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported prompt_version '{prompt_version}'. Supported: {supported}"
        )
    return builder(outcome_type=outcome_type, amount=amount)


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", "")
            if content_type in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    parts.append(text)

    return "\n".join(parts).strip()


def _is_retryable_error(error: Exception) -> bool:
    """Return True for transient API failures that should be retried."""
    retryable_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }
    if error.__class__.__name__ in retryable_names:
        return True

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500

    return False


def call_model(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
) -> Tuple[str, str]:
    """Call the Responses API with retry/backoff and return text + request id."""
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                temperature=temperature,
            )
            response_text = _extract_response_text(response)
            request_id = str(
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
                or ""
            )
            return response_text, request_id
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not _is_retryable_error(error):
                raise

            backoff_seconds = 2 ** attempt
            time.sleep(backoff_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("call_model failed without a captured exception")


def parse_response(response_text: str) -> Tuple[Optional[int], bool, bool, str]:
    """Parse model output into wager + validity + refusal flag + parse error type."""
    match = WAGER_PATTERN.search(response_text)
    parsed_wager = int(match.group(0)) if match else None
    valid = parsed_wager is not None

    lower_text = response_text.lower()
    has_refusal_language = any(marker in lower_text for marker in REFUSAL_MARKERS)

    # Refusal is tracked independently from numeric parsing validity.
    refusal_flag = has_refusal_language

    if not response_text.strip():
        parse_error_type = "empty_response"
    elif has_refusal_language:
        parse_error_type = "refusal_language"
    elif not valid:
        parse_error_type = "no_number"
    else:
        parse_error_type = ""

    return parsed_wager, valid, refusal_flag, parse_error_type


def _normalize_temperature(value: Any) -> str:
    """Normalize temperature to a stable string for resume keys."""
    try:
        return format(float(value), ".12g")
    except (TypeError, ValueError):
        return ""


def _load_resume_state(
    results_path: Path,
) -> Tuple[Set[Tuple[str, int, str, str, str]], int]:
    """Read existing results to determine completed runs and next trial_id."""
    completed_runs: Set[Tuple[str, int, str, str, str]] = set()
    max_trial_id = 0

    if not results_path.exists() or results_path.stat().st_size == 0:
        return completed_runs, max_trial_id

    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = (row.get("condition") or "").strip()

            try:
                run_number = int(row.get("run_number", "0"))
            except ValueError:
                run_number = 0

            model = (row.get("model") or "").strip()
            temp_key = _normalize_temperature(row.get("temperature"))
            prompt_version = (row.get("prompt_version") or "").strip()

            try:
                trial_id = int(row.get("trial_id", "0"))
            except ValueError:
                trial_id = 0

            if condition and run_number > 0 and model and temp_key and prompt_version:
                completed_runs.add((condition, run_number, model, temp_key, prompt_version))

            max_trial_id = max(max_trial_id, trial_id)

    return completed_runs, max_trial_id


def run_experiment(
    conditions_path: Path,
    output_path: Path,
    n_trials: int,
    model: str,
    temperature: float,
    sleep_seconds: float = 0.0,
    prompt_version: str = "absolute",
    shuffle: bool = False,
    max_retries: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Execute the experiment and append each row immediately to disk."""
    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or a positive integer")
    if not math.isfinite(temperature) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a finite number between 0 and 2")

    conditions = load_conditions(conditions_path)
    if seed is not None:
        random.seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_runs, max_trial_id = _load_resume_state(output_path)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    client = OpenAI()
    trial_id = max_trial_id
    total_new = 0
    temp_key = _normalize_temperature(temperature)

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        trial_plan: List[Tuple[Dict[str, object], int]] = []
        for condition in conditions:
            for run_number in range(1, n_trials + 1):
                trial_plan.append((condition, run_number))

        if shuffle:
            random.shuffle(trial_plan)

        for condition, run_number in trial_plan:
            condition_name = str(condition["condition"])
            outcome_type = str(condition["outcome_type"])
            amount = int(condition["amount"])

            run_key = (condition_name, run_number, model, temp_key, prompt_version)
            if run_key in completed_runs:
                continue

            prompt_text = build_prompt(
                outcome_type=outcome_type,
                amount=amount,
                prompt_version=prompt_version,
            )
            response_text, request_id = call_model(
                client=client,
                prompt=prompt_text,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )
            parsed_wager, valid, refusal_flag, parse_error_type = parse_response(response_text)
            log_wager = math.log(parsed_wager) if valid and parsed_wager is not None else ""

            trial_id += 1
            row = {
                "trial_id": trial_id,
                "condition": condition_name,
                "outcome_type": outcome_type,
                "amount": amount,
                "run_number": run_number,
                "model": model,
                "temperature": temperature,
                "prompt_version": prompt_version,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "parsed_wager": parsed_wager if parsed_wager is not None else "",
                "log_wager": log_wager,
                "valid": valid,
                "refusal_flag": refusal_flag,
                "parse_error_type": parse_error_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
            }

            # Persist every trial row so interrupted runs can be resumed safely.
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

            completed_runs.add(run_key)
            total_new += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Completed. Added {total_new} new trial rows to {output_path}.")


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    parser = argparse.ArgumentParser(description="Run realization-effect LLM experiment.")
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("conditions.csv"),
        help="Path to conditions CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/results.csv"),
        help="Path to output results CSV file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of runs per condition (default: 100).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to the API (default: 1.0).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="absolute",
        help="Prompt wording version (default: absolute).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle condition order before running.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient API failures (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible shuffling.",
    )
    args = parser.parse_args()

    run_experiment(
        conditions_path=args.conditions,
        output_path=args.output,
        n_trials=args.n_trials,
        model=args.model,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
        prompt_version=args.prompt_version,
        shuffle=args.shuffle,
        max_retries=args.max_retries,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
