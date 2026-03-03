# chains.py

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from config import MODEL_NAME, TEMPERATURE, SEED, logger
from schemas import intent_parser, intent_summary_parser, intent_summary_explain_parser
from prompts import intent_prompt, intent_summary_prompt, intent_summary_explain_prompt, rag_prompt
import os

logger.info("Initializing model: %s (temp=%s, seed=%s)", MODEL_NAME, TEMPERATURE, SEED)
model = OllamaLLM(model=MODEL_NAME, temperature=TEMPERATURE, seed=SEED)

# Intent classification chains (from Level 3)
intent_only_chain = intent_prompt | model | intent_parser
intent_only_chain.name = "intent_only"

intent_summary_chain = intent_summary_prompt | model | intent_summary_parser
intent_summary_chain.name = "intent_summary"

intent_summary_explain_chain = intent_summary_explain_prompt | model | intent_summary_explain_parser
intent_summary_explain_chain.name = "intent_summary_explain"

logger.debug("Intent classification chains created: %s, %s, %s", 
             intent_only_chain.name, 
             intent_summary_chain.name, 
             intent_summary_explain_chain.name)

# NEW: RAG chain for document-grounded Q&A
# Only initialize if vector store exists
if os.path.exists("vector_store"):
    from retriever import get_retriever
    
    logger.info("Vector store found - initializing RAG chain")
    retriever = get_retriever(k=3)
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",  # "stuff" = concat all retrieved chunks into one prompt
        retriever=retriever,
        return_source_documents=True,  # include which chunks were used
        chain_type_kwargs={"prompt": rag_prompt}
    )
    rag_chain.name = "rag_qa"
    
    logger.debug("RAG chain created with retriever (k=3)")
else:
    rag_chain = None
    logger.warning("Vector store not found - RAG chain unavailable. Run 'python indexing.py' to create it.")
