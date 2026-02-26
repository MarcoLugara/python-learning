# evaluation.py

from config import MAX_ATTEMPTS, logger
from chains import intent_only_chain, intent_summary_chain, intent_summary_explain_chain

EVAL_DATASET = [
    {"text": "My invoice shows a charge I don't recognize.", "expected_intent": "billing"},
    {"text": "My internet connection keeps dropping every few hours.", "expected_intent": "tech_support"},
    {"text": "I'm interested in upgrading to a business package.", "expected_intent": "sales"},
    {"text": "The green bond market has grown rapidly...", "expected_intent": "none"},
    {"text": "I was overcharged for the premium tier last cycle.", "expected_intent": "billing"},
    {"text": "How do I reset my router to factory settings?", "expected_intent": "tech_support"},
]

eval_chains = [
    intent_only_chain,
    intent_summary_chain,
    intent_summary_explain_chain,
]


def evaluate_chain(chains, dataset):
    """Evaluate multiple chains against a dataset and return minimum accuracy."""
    min_accuracy = 100.0
    total = len(dataset)

    for chain in chains:
        correct = 0
        logger.info("Evaluating chain: %s", chain.name)

        for item in dataset:
            result = chain.invoke({"text": item["text"]})
            predicted = result.get("intent")
            expected = item["expected_intent"]

            is_correct = predicted == expected
            correct += int(is_correct)

            status = "✔" if is_correct else "✖"
            logger.debug("%s  Expected: %s | Got: %s", status, expected, predicted)

        accuracy = correct / total * 100
        logger.info("Chain %s accuracy: %d/%d = %.1f%%", chain.name, correct, total, accuracy)
        min_accuracy = min(accuracy, min_accuracy)

    return min_accuracy


def run_evaluation():
    """Run evaluation loop with retry logic."""
    attempt = 0
    accuracy = 0.0

    while accuracy < 100 and attempt < MAX_ATTEMPTS:
        attempt += 1
        logger.info("Evaluation attempt %d/%d", attempt, MAX_ATTEMPTS)
        accuracy = evaluate_chain(eval_chains, EVAL_DATASET)

    if accuracy == 100.0:
        logger.info("✔ Evaluation passed with 100%% accuracy")
    else:
        logger.error("✖ Evaluation failed after %d attempts. Final accuracy: %.1f%%",
                     MAX_ATTEMPTS, accuracy)
        logger.error("Prompt or few-shot examples need manual review before proceeding.")
        raise SystemExit(1)