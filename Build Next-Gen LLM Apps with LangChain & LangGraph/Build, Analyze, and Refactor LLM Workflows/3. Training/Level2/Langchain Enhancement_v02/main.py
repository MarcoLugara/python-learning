# main.py

import asyncio
import time
from chains import intent_only_chain, intent_summary_chain, intent_summary_explain_chain, rag_chain
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
        ("intent_summary",         intent_summary_chain),
        ("intent_only",            intent_only_chain),
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


async def run_rag_query(question: str):
    """
    Run a RAG query to answer a question based on company documents.
    
    Args:
        question: User's question
        
    Returns:
        Dict with 'result' (answer) and 'source_documents' (chunks used)
    """
    if rag_chain is None:
        logger.error("RAG chain not available - run 'python indexing.py' first")
        return None
    
    logger.info("RAG Query: %s", question)
    
    # RAG chains use synchronous invoke, run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, rag_chain, {"query": question})
    
    return result


async def main():
    """Main async entry point."""
    start = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info("LangChain Intent Analysis + Document Grounding (RAG) - Level 3")
    logger.info("=" * 80)
    
    # Gate: evaluation must pass before any real analysis runs
    logger.info("\n[PHASE 1] Running Intent Classification Evaluation")
    logger.info("-" * 80)
    run_evaluation()
    
    # Phase 2: Intent classification on sample text
    logger.info("\n[PHASE 2] Intent Classification on Sample Text")
    logger.info("-" * 80)
    
    sample_text = """
The green bond market has grown rapidly over the past decade, reflecting a broader 
shift toward sustainable finance. Green bonds are debt instruments specifically issued 
to fund projects with environmental benefits — such as renewable energy installations, 
climate-resilient infrastructure, or low-carbon transportation.
    """
    
    logger.info("Processing sample text for intent classification...")
    result, chain_name = await progressive_intent_analysis(sample_text)
    
    logger.info("Intent classification result:")
    logger.info("  Chain used: %s", chain_name)
    logger.info("  Intent: %s", result.get("intent", "N/A"))
    
    # Phase 3: RAG Q&A if vector store exists
    if rag_chain is not None:
        logger.info("\n[PHASE 3] Document-Grounded Q&A (RAG)")
        logger.info("-" * 80)
        
        # Ask multiple questions to demonstrate retrieval
        questions = [
            "What is the return policy for defective products?",
            "How do I reset my router if it keeps disconnecting?",
            "What payment methods do you accept?",
            "What are the support hours for enterprise customers?",
        ]
        
        for question in questions:
            rag_result = await run_rag_query(question)
            
            if rag_result:
                answer = rag_result["result"]
                sources = rag_result.get("source_documents", [])
                
                logger.info("\nQ: %s", question)
                logger.info("A: %s", answer)
                logger.info("Sources used: %d document chunks", len(sources))
                
                # Log source metadata
                for i, doc in enumerate(sources, 1):
                    source_file = doc.metadata.get("source", "unknown")
                    logger.debug("  Source %d: %s (length: %d chars)", 
                               i, source_file, len(doc.page_content))
    else:
        logger.warning("\n[PHASE 3 SKIPPED] Vector store not found")
        logger.warning("Run 'python indexing.py' to enable document-grounded Q&A")
    
    elapsed = time.perf_counter() - start
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Execution Summary")
    logger.info("=" * 80)
    logger.info("Total execution time: %.2f seconds", elapsed)
    logger.info("Intent classification: %s", "✔ Complete")
    logger.info("RAG Q&A: %s", "✔ Complete" if rag_chain else "⊘ Skipped (no vector store)")
    logger.info("=" * 80)
    
    # Persist intent classification result
    save_result(result, chain_name, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
