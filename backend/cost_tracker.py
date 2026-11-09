import os
import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
COST_LOG = BASE_DIR / "experiments" / "cost_logs.jsonl"

load_dotenv(BASE_DIR / ".env")

INPUT_PRICE = float(os.getenv("GEMINI_INPUT_PRICE_PER_MILLION", "0"))
OUTPUT_PRICE = float(os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION", "0"))


def extract_token_usage(response):
    """
    Extract token usage from a LangChain AIMessage.
    """

    usage = getattr(response, "usage_metadata", None)

    if not usage:
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if input_tokens is None or output_tokens is None:
        return None

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


def calculate_cost(input_tokens, output_tokens):
    input_cost = (
        input_tokens * INPUT_PRICE / 1000000
    )

    output_cost = (
        output_tokens * OUTPUT_PRICE / 1000000
    )

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }


def track_cost(response, model_name):
    """
    Record token usage and estimated API cost.
    """

    usage = extract_token_usage(response)

    if usage is None:
        print("WARNING: Token usage metadata unavailable.")
        return None

    cost = calculate_cost(
        usage["input_tokens"],
        usage["output_tokens"]
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        **usage,
        **cost
    }

    COST_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(COST_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    print(
        f"Estimated API cost: ${cost['total_cost']:.8f}"
    )

    return record


def get_cost_summary():
    """
    Calculate cumulative API spending from saved records.
    """

    if not COST_LOG.exists():
        return {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_cost_per_request": 0.0
        }

    records = []

    with open(COST_LOG, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    total_requests = len(records)

    total_input = sum(
        record["input_tokens"] for record in records
    )

    total_output = sum(
        record["output_tokens"] for record in records
    )

    total_cost = sum(
        record["total_cost"] for record in records
    )

    return {
        "total_requests": total_requests,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost": total_cost,
        "average_cost_per_request": (
            total_cost / total_requests
            if total_requests else 0.0
        )
    }