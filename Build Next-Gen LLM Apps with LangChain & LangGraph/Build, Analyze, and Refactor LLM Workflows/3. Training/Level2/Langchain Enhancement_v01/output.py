# output.py

import json
from datetime import datetime, timezone
from config import logger

OUTPUT_FILE = "results.jsonl"


def save_result(result: dict, chain_name: str, execution_time: float) -> None:
    """
    Persist result to JSONL file with metadata.

    Args:
        result: The classification result dictionary
        chain_name: Name of the chain that produced the result
        execution_time: Total execution time in seconds
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain_used": chain_name,
        "execution_time_seconds": round(execution_time, 2),
        "result": result,
    }

    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Result written to %s (chain: %s, time: %.2fs)",
                    OUTPUT_FILE, chain_name, execution_time)
    except IOError as e:
        logger.error("Failed to write result to file: %s", e)