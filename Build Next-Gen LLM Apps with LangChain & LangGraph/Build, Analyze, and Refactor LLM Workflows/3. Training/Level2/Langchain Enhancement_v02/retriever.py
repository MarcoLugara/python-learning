# retriever.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import logger

VECTOR_STORE_DIR = "vector_store"

def get_retriever(k=3):
    """
    Load the vector store and return a retriever.
    
    The retriever searches for semantically similar document chunks
    based on the input query.
    
    Args:
        k: Number of most relevant chunks to retrieve (default: 3)
        
    Returns:
        A LangChain retriever object with .get_relevant_documents(query) method
    """
    logger.debug("Loading vector store from %s", VECTOR_STORE_DIR)
    
    # Use the same embedding model as indexing
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Load the persisted Chroma vector store
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings
    )
    
    # Create retriever with configurable k
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    logger.debug("Retriever created (k=%d, search_type=similarity)", k)
    
    return retriever
