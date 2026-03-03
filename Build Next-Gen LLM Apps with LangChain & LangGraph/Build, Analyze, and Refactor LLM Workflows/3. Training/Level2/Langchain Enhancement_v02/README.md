# LangChain Intent Analysis + Document Grounding (RAG)

## Project Structure

```
level3_rag/
├── documents/              # Source documents for RAG
│   ├── return_policy.txt
│   ├── tech_support.txt
│   └── billing_info.txt
├── vector_store/          # Generated: Chroma vector database (after running indexing.py)
├── config.py              # Configuration and logger setup
├── schemas.py             # Pydantic models and output parsers
├── prompts.py             # Prompt templates (intent + RAG)
├── chains.py              # Chain assembly (intent + RAG)
├── evaluation.py          # Evaluation dataset and scoring
├── indexing.py            # Vector store builder (run once)
├── retriever.py           # Vector store loader for queries
├── output.py              # JSONL result persistence
├── main.py                # Async orchestration (intent + RAG demo)
├── run.log                # Generated: structured logs
└── results.jsonl          # Generated: persisted results
```

## What This Demonstrates

### Level 3 Features (from previous iteration)
1. **Async Parallel Execution** — All intent classification chains run simultaneously
2. **Structured Logging** — Professional observability with dual console/file output
3. **Output Persistence** — Results saved to JSONL for audit trail

### NEW: Document Grounding (RAG)
4. **Vector Store Indexing** — Documents chunked, embedded, stored in Chroma
5. **Semantic Retrieval** — Finds relevant chunks based on query meaning
6. **Grounded Generation** — LLM answers based on retrieved company documents
7. **Source Attribution** — Shows which document chunks were used

## Complete Setup Instructions

### 1. Install Dependencies

```bash
# Core dependencies (if not already installed)
pip install langchain langchain-community langchain-ollama --break-system-packages

# NEW: RAG dependencies
pip install chromadb sentence-transformers --break-system-packages
```

### 2. Verify Ollama is Running

```bash
# Make sure Ollama is running with phi3:mini available
ollama list
ollama pull phi3:mini  # if not present
```

### 3. Build the Vector Store (CRITICAL - Run This First)

```bash
python indexing.py
```

**What this does:**
- Loads all `.txt` files from `documents/` folder
- Splits them into 500-token chunks with 100-token overlap
- Embeds each chunk using `all-MiniLM-L6-v2` (runs locally, no API key)
- Stores embeddings in `vector_store/` directory using Chroma

**Expected output:**
```
2026-02-24 14:23:01 | INFO | indexing | Loading documents from documents
2026-02-24 14:23:01 | INFO | indexing | Loaded 3 documents
2026-02-24 14:23:02 | INFO | indexing | Split into 18 chunks
2026-02-24 14:23:02 | INFO | indexing | Initializing embedding model: all-MiniLM-L6-v2
2026-02-24 14:23:15 | INFO | indexing | Building vector store...
2026-02-24 14:23:45 | INFO | indexing | ✔ Vector store created and persisted to vector_store
```

**You only need to run this:**
- Once initially
- When you add/modify documents in `documents/` folder

### 4. Run the Main Application

```bash
python main.py
```

**What this does:**
- **Phase 1:** Runs evaluation on intent classification chains
- **Phase 2:** Demonstrates async parallel intent classification
- **Phase 3:** Runs 4 RAG queries against your company documents

## Expected Output

### Console Output (INFO level)

```
================================================================================
LangChain Intent Analysis + Document Grounding (RAG) - Level 3
================================================================================

[PHASE 1] Running Intent Classification Evaluation
--------------------------------------------------------------------------------
2026-02-24 14:25:01 | INFO | evaluation | Evaluating chain: intent_only
2026-02-24 14:25:15 | INFO | evaluation | Chain intent_only accuracy: 6/6 = 100.0%
2026-02-24 14:25:15 | INFO | evaluation | ✔ Evaluation passed with 100% accuracy

[PHASE 2] Intent Classification on Sample Text
--------------------------------------------------------------------------------
2026-02-24 14:25:16 | INFO | main | Starting parallel execution of 3 chains
2026-02-24 14:25:42 | INFO | main | ✔ Success using intent_summary_explain chain
2026-02-24 14:25:42 | INFO | main | Intent classification result:
2026-02-24 14:25:42 | INFO | main |   Chain used: intent_summary_explain
2026-02-24 14:25:42 | INFO | main |   Intent: none

[PHASE 3] Document-Grounded Q&A (RAG)
--------------------------------------------------------------------------------
2026-02-24 14:25:43 | INFO | main | RAG Query: What is the return policy for defective products?
2026-02-24 14:25:48 | INFO | main | 
Q: What is the return policy for defective products?
A: For defective products, we offer free return shipping and a full refund or replacement. Customers must report defects within 14 days of delivery. We will provide a prepaid shipping label for the return.
2026-02-24 14:25:48 | INFO | main | Sources used: 3 document chunks

2026-02-24 14:25:49 | INFO | main | RAG Query: How do I reset my router if it keeps disconnecting?
2026-02-24 14:25:53 | INFO | main | 
Q: How do I reset my router if it keeps disconnecting?
A: If your router loses connection frequently, first try a factory reset by holding the reset button for 10 seconds. Then reconfigure using the setup wizard in the admin panel at 192.168.1.1. If issues persist, check for firmware updates.
2026-02-24 14:25:53 | INFO | main | Sources used: 3 document chunks

================================================================================
Execution Summary
================================================================================
Total execution time: 52.35 seconds
Intent classification: ✔ Complete
RAG Q&A: ✔ Complete
================================================================================
```

### `run.log` (includes DEBUG details)

Same as console, plus:
- Detailed chunk retrieval information
- Source file paths for each retrieved chunk
- Character counts and metadata

### `results.jsonl`

```json
{"timestamp": "2026-02-24T14:25:42.123456+00:00", "chain_used": "intent_summary_explain", "execution_time_seconds": 52.35, "result": {"summary_of_the_text": "...", "intent": "none", "justification_of_choice": "..."}}
```

## How RAG Works - Technical Flow

### Indexing Phase (offline)
```
documents/return_policy.txt
    ↓ (load)
"For defective products, we offer free return shipping..."
    ↓ (split into 500-token chunks)
Chunk 1: "For defective products..."
Chunk 2: "Enterprise customers have..."
    ↓ (embed each chunk)
[0.23, 0.81, -0.45, ...] (1536 dimensions)
    ↓ (store in Chroma)
vector_store/ database
```

### Query Phase (runtime)
```
User: "What's the return policy for defective items?"
    ↓ (embed query)
[0.24, 0.79, -0.43, ...] (same 1536 dimensions)
    ↓ (similarity search in Chroma)
Top 3 chunks:
  1. "For defective products, we offer..." (similarity: 0.92)
  2. "Customers must report defects within..." (similarity: 0.87)
  3. "Enterprise customers have extended..." (similarity: 0.81)
    ↓ (concat chunks + query into prompt)
Prompt: "Context: [chunk1][chunk2][chunk3] Question: ..."
    ↓ (LLM generates answer)
"For defective products, we offer free return shipping and..."
```

## Key Parameters to Tune

### In `indexing.py`:
- **chunk_size** (500) — Larger = more context per chunk, fewer chunks
- **chunk_overlap** (100) — Prevents losing context at boundaries

### In `retriever.py`:
- **k** (3) — Number of chunks to retrieve. Higher = more context but slower

### In `chains.py`:
- **chain_type** ("stuff") — How chunks are combined:
  - "stuff" = concat all into one prompt (fast, limited by context window)
  - "map_reduce" = summarize each chunk, then combine (slower, handles more chunks)
  - "refine" = iterative refinement (slowest, highest quality)

## Adding Your Own Documents

1. Add `.txt` files to `documents/` folder
2. Re-run: `python indexing.py`
3. Run: `python main.py`

**Supported formats** (with appropriate loaders):
- `.txt` — plain text
- `.pdf` — requires `pypdf` package
- `.docx` — requires `python-docx` package
- `.html` — requires `beautifulsoup4` package

## Dependencies Summary

```bash
# Core LangChain
langchain
langchain-community
langchain-ollama

# Vector store and embeddings
chromadb
sentence-transformers

# Optional: for other document types
pypdf           # for PDF support
python-docx     # for Word docs
beautifulsoup4  # for HTML
```

## Architecture Comparison

| Feature | Level 2 | Level 3 | Level 3 + RAG |
|---------|---------|---------|---------------|
| Intent classification | ✓ | ✓ | ✓ |
| Sequential execution | ✓ | - | - |
| Parallel execution | - | ✓ | ✓ |
| Print statements | ✓ | - | - |
| Structured logging | - | ✓ | ✓ |
| JSONL persistence | - | ✓ | ✓ |
| Document grounding | - | - | ✓ |
| Vector search | - | - | ✓ |
| Source attribution | - | - | ✓ |

## Next Steps

To extend this further:
1. **Add metadata filtering** — Tag chunks by department, retrieve only relevant ones
2. **Hybrid search** — Combine vector search with keyword search (BM25)
3. **Re-ranking** — Retrieve 10 chunks, re-rank to top 3 for precision
4. **Conversational memory** — Add chat history to context
5. **Multi-query retrieval** — Generate multiple search queries for one question
6. **Evaluation dataset** — Build (question, expected_answer, expected_sources) test set

## Troubleshooting

**"Vector store not found" warning:**
- Run `python indexing.py` first

**"No module named 'chromadb'":**
- Run `pip install chromadb sentence-transformers --break-system-packages`

**Embedding model download slow:**
- First run downloads `all-MiniLM-L6-v2` (~80MB). Subsequent runs are instant.

**RAG answers are inaccurate:**
- Check retrieved chunks with `logger.debug` — are the right chunks being found?
- Try increasing `k` (retrieve more chunks)
- Try larger `chunk_size` (more context per chunk)
- Improve document quality (clearer, better structured source docs)
