# main.py

import json
import time
from chains import intent_only_chain, intent_summary_chain, intent_summary_explain_chain
from evaluation import run_evaluation


def progressive_intent_analysis(text):
    chains = [
        ("intent_summary_explanation", intent_summary_explain_chain),
        ("intent_summary",             intent_summary_chain),
        ("intent_only",                intent_only_chain),
    ]
    for label, chain in chains:
        try:
            result = chain.invoke({"text": text})
            print(f"✔ Success using {label} chain")
            return result
        except Exception as e:
            print(f"✖ {label} chain failed: {e}")

    return {"error": "All fallback chains failed"}


if __name__ == "__main__":
    start = time.perf_counter()

    # Gate: evaluation must pass before any real analysis runs
    run_evaluation()

    sample_text = """
    The green bond market has grown rapidly over the past decade...
    """

    result  = progressive_intent_analysis(sample_text)
    elapsed = time.perf_counter() - start

    print(json.dumps(result, indent=2))
    print(f"Chain took {elapsed:.2f}s")