# main.py

import asyncio
import time
from chains import intent_only_chain, intent_summary_chain, intent_summary_explain_chain
from evaluation import run_evaluation
from output import save_result
from config import logger


async def progressive_intent_analysis(text: str):
    """
    Run all three chains in parallel and return the first successful result.

    Uses asyncio.gather with return_exceptions=True to run chains concurrently.
    Falls back through results in priority order: explain → summary → intent_only.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (result_dict, chain_name) or (error_dict, "error")
    """
    chains = [
        ("intent_summary_explain", intent_summary_explain_chain),
        ("intent_summary", intent_summary_chain),
        ("intent_only", intent_only_chain),
    ]

    logger.info("Starting parallel execution of %d chains", len(chains))
    logger.debug("Input text length: %d characters", len(text))

    # Launch all chains simultaneously
    tasks = [chain.ainvoke({"text": text}) for label, chain in chains]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results in priority order (same order as chains list)
    for (label, chain), result in zip(chains, results):
        if isinstance(result, Exception):
            logger.warning("Chain %s failed: %s", label, str(result))
            continue

        # First successful result wins
        logger.info("✔ Success using %s chain", label)
        return result, label

    # All chains failed
    logger.error("✖ All chains failed")
    return {"error": "All fallback chains failed"}, "error"


async def main():
    """Main async entry point."""
    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Starting LangChain Intent Analysis - Level 3")
    logger.info("=" * 60)

    # Gate: evaluation must pass before any real analysis runs
    run_evaluation()

    sample_text = """
The green bond market has grown rapidly over the past decade, reflecting a broader 
shift toward sustainable finance. Green bonds are debt instruments specifically issued 
to fund projects with environmental benefits — such as renewable energy installations, 
climate-resilient infrastructure, or low-carbon transportation. Governments, corporations, 
and financial institutions all participate, often using independent verification to 
reassure investors that the funds are genuinely directed toward sustainability goals.

For investors, green bonds offer a way to combine financial returns with environmental 
impact. Yields are typically comparable to conventional bonds from similar issuers, but 
demand is often strong because institutional investors — including pension funds and asset 
managers — increasingly have sustainability mandates. This demand can sometimes allow 
issuers to secure slightly better financing terms.

Despite the optimism, the sector still faces challenges. Standardization of reporting, 
prevention of "greenwashing," and consistent impact measurement remain ongoing concerns. 
Even so, most analysts expect the green bond segment to remain a central pillar of the 
evolving sustainable finance landscape.
    """

    logger.info("Processing sample text...")
    result, chain_name = await progressive_intent_analysis(sample_text)

    elapsed = time.perf_counter() - start

    # Log final result
    logger.info("=" * 60)
    logger.info("Final result from chain '%s':", chain_name)
    logger.info("Intent: %s", result.get("intent", "N/A"))
    if "summary_of_the_text" in result:
        logger.info("Summary: %s", result["summary_of_the_text"])
    if "justification_of_choice" in result:
        logger.info("Justification: %s", result["justification_of_choice"])
    logger.info("Total execution time: %.2f seconds", elapsed)
    logger.info("=" * 60)

    # Persist to JSONL
    save_result(result, chain_name, elapsed)


if __name__ == "__main__":
    asyncio.run(main())