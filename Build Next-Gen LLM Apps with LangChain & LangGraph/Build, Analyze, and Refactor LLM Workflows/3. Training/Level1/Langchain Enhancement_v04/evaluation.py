# evaluation.py

from config import MAX_ATTEMPTS
from chains import intent_only_chain, intent_summary_chain, intent_summary_explain_chain

EVAL_DATASET = [
    {"text": "My invoice shows a charge I don't recognize.",  "expected_intent": "billing"},
    {"text": "My internet connection keeps dropping every few hours.", "expected_intent": "tech_support"},
    {"text": "I'm interested in upgrading to a business package.", "expected_intent": "sales"},
    {"text": "The green bond market has grown rapidly...",          "expected_intent": "none"},
    {"text": "I was overcharged for the premium tier last cycle.",  "expected_intent": "billing"},
    {"text": "How do I reset my router to factory settings?",      "expected_intent": "tech_support"}
]

eval_chains = [
    intent_only_chain,
    intent_summary_chain,
    intent_summary_explain_chain
]

def evaluate_chain(chains, dataset):
    min_accuracy = 100.0
    total = len(dataset)

    for chain in chains:
        correct: int = 0
        print(f" We run chain {chain.name} for our evaluation dataset.")
        for item in dataset:
            result    = chain.invoke({"text": item["text"]})
            predicted = result.get("intent")
            expected  = item["expected_intent"]

            is_correct = predicted == expected
            correct   += int(is_correct)

            status = "✔" if is_correct else "✖"
            print(f"{status}  Expected: {expected:12} | Got: {predicted}")

        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
        min_accuracy = min(accuracy, min_accuracy)

    return min_accuracy


def run_evaluation():
    attempt  = 0
    accuracy = 0.0

    while accuracy < 100 and attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n--- Evaluation attempt {attempt}/{MAX_ATTEMPTS} ---")
        accuracy = evaluate_chain(eval_chains, EVAL_DATASET)

    if accuracy == 100.0:
        print("✔ Evaluation passed.")
    else:
        print(f"✖ Evaluation failed after {MAX_ATTEMPTS} attempts. Final accuracy: {accuracy:.1f}%")
        print("  Prompt or few-shot examples need manual review before proceeding.")
        raise SystemExit(1)