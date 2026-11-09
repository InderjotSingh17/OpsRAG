
import json
import threading
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "experiments" / "observability.jsonl"

_log_lock = threading.Lock()


def log_event(event_type, request_id, **details):
    """Write one structured observability event."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "request_id": request_id,
        **details
    }

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with _log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

    except Exception as error:
        print(f"Observability logging failed: {type(error).__name__}")


def get_observability_summary():
    """Aggregate successful, rejected and failed requests."""

    summary = {
        "total_requests": 0,
        "successful_requests": 0,
        "rejected_requests": 0,
        "failed_requests": 0,
        "average_latency_seconds": 0.0
    }

    if not LOG_FILE.exists():
        return summary

    request_ids = set()
    terminal_events = {}

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                request_id = record.get("request_id")
                event = record.get("event")

                if not request_id:
                    continue

                request_ids.add(request_id)

                if event in (
                    "request_completed",
                    "request_failed",
                    "guardrail_rejected"
                ):
                    terminal_events[request_id] = record

    except OSError as error:
        print(f"Observability summary failed: {type(error).__name__}")
        return summary

    summary["total_requests"] = len(request_ids)

    latencies = []

    for record in terminal_events.values():
        event = record.get("event")

        if event == "request_completed":
            if record.get("status") == "success":
                summary["successful_requests"] += 1
            else:
                summary["failed_requests"] += 1

        elif event == "guardrail_rejected":
            summary["rejected_requests"] += 1

        elif event == "request_failed":
            
            if record.get("stage") == "input_validation":
                summary["rejected_requests"] += 1
            else:
                summary["failed_requests"] += 1

        latency = record.get("total_latency_seconds")

        if isinstance(latency, (int, float)):
            latencies.append(latency)

    if latencies:
        summary["average_latency_seconds"] = round(
            sum(latencies) / len(latencies), 4
        )

    return summary
