# indexing.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
'''
    DirectoryLoader — A loader that scans a directory for files matching a pattern
    TextLoader — A loader that reads a single text file and creates a Document object from it
'''

from langchain_text_splitters import RecursiveCharacterTextSplitter
#   RecursiveCharacterTextSplitter - Imports the text splitting class that breaks documents into chunks

from langchain_community.embeddings import HuggingFaceEmbeddings
'''
    What this does: Imports the embedding model wrapper.
    What embeddings are: A way to convert text into a vector of numbers (typically 384 or 768 
     or 1536 dimensions) where semantically similar text has similar vectors. Example:

    "cat" → [0.2, 0.8, -0.1, 0.5, ...]
    "kitten" → [0.21, 0.79, -0.09, 0.51, ...] (very close)
    "database" → [-0.6, 0.1, 0.9, -0.3, ...] (far away)
    
    HuggingFaceEmbeddings wraps models from Hugging Face that do this conversion
'''

from langchain_community.vectorstores import Chroma
'''
    What this does: Imports Chroma, the vector database.
    What a vector database does: Stores vectors and lets you search for "which vectors are most 
     similar to this query vector?" Very different from SQL databases that do exact matches. 
    Chroma uses HNSW (Hierarchical Navigable Small World) algorithm internally for fast 
     approximate nearest neighbor search.
'''

from config import logger   #see config.py

DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"

def build_vector_store():
    """
    Load documents from DOCUMENTS_DIR, split into chunks, embed, and store in Chroma.
    
    This should be run once initially, and again whenever documents are updated.
    """
    logger.info("=" * 60)
    logger.info("Building Vector Store for Document Grounding")
    logger.info("=" * 60)
    '''
    These are INFO level logger messages. These appear on console (INFO+) 
    and in run.log (DEBUG+), see config.py
    '''

    # Load all text files from documents directory
    logger.info("Loading documents from %s", DOCUMENTS_DIR)
    loader = DirectoryLoader(
        DOCUMENTS_DIR, 
        glob="**/*.txt",  #glob means "any subdirectory depth", namely with *=anything
        loader_cls=TextLoader,   #for each matched file, use TextLoader to read it
        loader_kwargs={"encoding": "utf-8"}
    # loader_kwargs — short for "loader keyword arguments", specifically telling it to read the file using UTF-8 encoding.
    )
# P.S. At this point: No files are read yet. This just configures the loader.

#Now, the loading actually happens
    documents = loader.load()
    '''
    Internally, DirectoryLoader does:
    1. Scans documents/ recursively
    2. Finds all .txt files
    3. For each file, creates a TextLoader("path/to/file.txt", encoding="utf-8")
    4. Calls .load() on each TextLoader (function of DirectoryLoader
    4. Each TextLoader reads the file and creates a Document object with:
       .page_content — the full text of the file
       .metadata — a dict like {"source": "documents/return_policy.txt"}
    5. Returns a list of all Document objects
    '''

    logger.info("Loaded %d documents", len(documents))
    # Logs how many document files were found.
    # If this is 0, something is wrong with the directory path.
    
    # Log document sources
    for doc in documents:
        logger.debug("Document source: %s (length: %d chars)", 
                    doc.metadata.get("source", "unknown"), 
                    len(doc.page_content))
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    
    # Initialize embedding model (runs locally, no API key needed)
    logger.info("Initializing embedding model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create vector store
    logger.info("Building vector store (this may take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    
    logger.info("✔ Vector store created and persisted to %s", VECTOR_STORE_DIR)
    logger.info("Total chunks indexed: %d", len(chunks))
    logger.info("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    # Run this script directly to build the index
    # python indexing.py
    build_vector_store()
