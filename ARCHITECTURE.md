# 🔬 Ollama Deep Researcher — Architecture & Reference Guide

> **For interview prep and onboarding. Generated from source on 2026-02-23.**

---

## 1. What Is This Project?

An enterprise AI research assistant that answers complex technical questions by simultaneously querying **Jira**, **Confluence**, and **Perforce** — then cross-referencing, summarizing, and fact-checking the results using **locally-running LLMs** via Ollama. The entire workflow is orchestrated as a stateful graph using **LangGraph**.

**Key differentiators:**
- Everything runs **locally** — no external AI APIs (uses Ollama + open-source models like `qwen3:30b`)
- Queries **three enterprise systems in parallel** (Jira, Confluence, Perforce)
- Has a **multi-stage anti-hallucination pipeline** (fact checking, self-verification, multi-agent verification)
- Uses **real hybrid search** (BM25 sparse + FAISS dense) for accurate document retrieval

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| LLM Runtime | Ollama (local), default model: `qwen3:30b-a3b` |
| Embeddings | `nomic-embed-text` via Ollama |
| Orchestration | LangGraph `0.4.3` (StateGraph) |
| Vector Store | FAISS (via LangChain community) |
| Sparse Search | BM25 via `rank-bm25` library (`BM25Okapi`) |
| Enterprise Tools | Jira SDK, Confluence REST API, Perforce P4Python |
| Web Search | DuckDuckGo + Tavily |
| Frontend | Next.js (in `frontend/`) |
| Language | Python 3.x, fully async |

---

## 3. Repository Structure

```
ollama-deep-researcher/
├── run_graph_3.py              ← Main entrypoint
├── requirements.txt            ← Python dependencies (incl. rank-bm25)
├── langgraph.json              ← LangGraph deployment config
├── pyproject.toml
├── .env.example                ← All environment variables documented here
├── README.md
├── frontend/                   ← Next.js chat UI
├── vector_store/               ← FAISS index stored on disk
└── src/
    └── assistant/
        ├── graph_3.py          ← THE MAIN GRAPH (all nodes & edges defined here)
        ├── state.py            ← ResearchState (shared dict passed to all nodes)
        ├── configuration.py    ← Singleton config (reads .env)
        ├── prompts.py          ← All LLM prompt templates (7 prompts)
        ├── register_graph.py   ← Registers graph with LangGraph server
        ├── rag/
        │   ├── ollama_retrieval.py    ← FAISS + Ollama embeddings (RAGRetriever)
        │   └── hybrid_retriever.py   ← BM25 + Dense + HyDE + Multi-query + RRF
        ├── analyzers/
        │   ├── cross_source_analyzer.py   ← Connects findings across sources
        │   ├── security_analyzer.py
        │   └── technical_analyzer.py
        ├── memory/
        │   └── knowledge_graph.py     ← Temporal knowledge graph
        ├── processors/
        │   ├── chunk_processor.py     ← Splits large documents into chunks
        │   └── jira_processor.py
        ├── research/
        │   ├── core.py, engine.py, search.py, analysis.py
        │   ├── graph.py, tools.py, types.py, utils.py
        │   └── error_handling.py, knowledge_graph.py
        ├── summarizers/
        │   ├── map_rerank.py              ← MapReduce with reranking
        │   ├── jira_summarizer.py
        │   ├── confluence_summarizer.py
        │   └── perforce_summarizer.py
        ├── tools/
        │   ├── PerforceTool.py        ← Full P4 client wrapper (~2800 lines)
        │   ├── jira_tool.py / jira_client.py / tool_jira.py
        │   ├── tool_confluence.py
        │   ├── tool_perforce.py
        │   └── SolutionBookTool.py
        ├── verifiers/
        │   ├── fact_checker.py              ← Claims extraction + verification
        │   ├── self_verification.py
        │   ├── multi_agent_verification.py  ← Multiple LLM agents cross-check
        │   ├── chain_of_knowledge.py
        │   ├── memory_augmented_verification.py
        │   └── tool_output_verification.py
        └── utils/
            ├── langgraph_helpers.py   ← process_in_batches, map_reduce_text, etc.
            ├── helpers.py
            ├── response_formatter.py
            └── ssl_fix.py
```

---

## 4. Main Graph — Data Flow (graph_3.py)

The system has **two connected LangGraph StateGraphs**:

### Graph 1: Main Research Pipeline

```
START
  ↓
process_query
  Expands user query with LLM (adds synonyms, MTV numbers, technical variants)
  ↓
retrieve_context
  Runs HybridRetriever: BM25 + FAISS + HyDE + multi-query expansion + RRF
  ↓
fetch_jira ──────────┐
fetch_perforce ──────┤  ← All three run in PARALLEL via LangGraph Send()
fetch_confluence ────┘
       ↓                ↓                  ↓
  process_jira   process_perforce   process_confluence
       ↓                ↓                  ↓
              summarize_data
         Merges all three sources into unified research summary
                     ↓
      apply_cross_source_analysis
         Finds connections between Jira ↔ Confluence ↔ Perforce findings
                     ↓
      apply_chain_of_knowledge
         Builds explicit reasoning chain (every conclusion traced to source)
                     ↓
      apply_map_rerank
         MapReduce: chunk → summarize → rerank by relevance → merge
                     ↓
      apply_multi_agent_verification
         3 independent LLM agents each evaluate the answer
                     ↓
      generate_final_summary
         Runs Fact Verification Sub-graph, then produces final answer with citations
                     ↓
                    END
```

### Graph 2: Fact Verification Sub-graph (nested inside generate_final_summary)

```
START → initial_verification → deep_verification
                                      ↓
                              need_revision?
                            yes ↓           no ↓
                       revise_content        END
                              ↓
                             END
```

---

## 5. RAG / Retrieval System (hybrid_retriever.py)

### Overview

Before querying live enterprise systems, the graph retrieves relevant **previously stored documents** from a local FAISS vector store. `HybridRetriever` combines 7 techniques:

```
User Query
    │
    ├─→ [1] BM25 Sparse Search        keyword exact match (BM25Index class)
    │
    ├─→ [2] Dense FAISS Search        semantic similarity via Ollama embeddings
    │
    ├─→ [3] Multi-Query Expansion     LLM generates 3-5 query variants
    │
    ├─→ [4] HyDE                      LLM writes hypothetical answer → embed → search
    │
    ├─→ [5] Step-Back Prompting       LLM generates a broader/abstract query
    │
    ├─→ [6] Recursive Retrieval       Extract key concepts from results → search again
    │
    └─→ All results merged via [7] RAG-Fusion (RRF)
            score = Σ 1 / (rank_in_list + k)
            Documents appearing high across multiple lists bubble to the top
```

### BM25Index class (sparse retrieval component)

```python
class BM25Index:
    def add_documents(documents: List[str])   # tokenize + index
    def search(query: str, top_k=10)          # returns (doc, score) pairs
    def _rebuild()                             # recomputes BM25 after adding new docs
    # Uses rank_bm25.BM25Okapi internally
    # Gracefully degrades if rank_bm25 is not installed
```

### Why BM25 + FAISS (not just FAISS)?

| Method | Good at | Bad at |
|---|---|---|
| FAISS (dense) | Semantic similarity, synonyms, concepts | Exact technical IDs like `MTV-1234` |
| BM25 (sparse) | Exact keyword matches, error codes, product names | Semantic understanding |
| Combined | Both — full spectrum coverage | — |

---

## 6. Configuration (configuration.py)

Singleton pattern — one instance shared across all graph nodes, reads from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `LOCAL_LLM` | qwen3:30b-a3b | LLM model name |
| `EMBEDDING_MODEL` | nomic-embed-text | Embedding model |
| `ENABLE_JIRA` | true | Toggle Jira source |
| `ENABLE_CONFLUENCE` | true | Toggle Confluence source |
| `ENABLE_PERFORCE` | true | Toggle Perforce source |
| `MAX_JIRA_RESULTS` | 50 | Result limit per query |
| `CHUNK_SIZE` | configurable | RAG chunk size |
| `CACHE_TTL` | 3600 | Cache TTL in seconds |

---

## 7. State (state.py)

`ResearchState` extends `Dict[str, Any]` — the shared object passed between all nodes:

| Field | Type | Description |
|---|---|---|
| `research_topic` | str | Original user question |
| `search_query` | str | Processed/expanded query |
| `running_summary` | str | Accumulated answer as graph runs |
| `raw_jira_results` | List[Dict] | Raw Jira API results |
| `raw_confluence_results` | List | Raw Confluence results |
| `raw_perforce_results` | List[Dict] | Raw Perforce results |
| `enterprise_research_results` | List[Dict] | Processed combined results |
| `sources_gathered` | List[str] | Source citations collected |
| `final_summary` | str | Final answer |
| `_errors` | List[Dict] | Error tracking |

---

## 8. Prompts (prompts.py)

| Name | Purpose |
|---|---|
| `query_writer_instructions` | Expand query → better search terms (handles MTV numbers) |
| `summarizer_instructions` | Summarize raw tool results into structured findings |
| `reflection_instructions` | LLM identifies gaps in current research |
| `security_analysis_instructions` | Detect security-relevant findings |
| `technical_validation_instructions` | Validate technical accuracy of claims |
| `knowledge_graph_instructions` | Extract entities and relationships |
| `citation_extraction_instructions` | Pull source citations from responses |

---

## 9. Anti-Hallucination Pipeline (5 layers)

| Layer | Module | What it does |
|---|---|---|
| 1 | `fact_checker.py` | Extracts individual claims → verifies each against source docs |
| 2 | `self_verification.py` | LLM checks its own output for logical consistency |
| 3 | `multi_agent_verification.py` | 3 independent LLM agents evaluate; disagreements flagged |
| 4 | `chain_of_knowledge.py` | Reasoning chain so every conclusion traces to a source |
| 5 | Fact Verification Sub-graph | Dedicated nested graph: initial check → deep check → revise loop |

---

## 10. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Jira/Confluence/Perforce credentials

# 3. Pull Ollama models
ollama pull qwen3:30b-a3b
ollama pull nomic-embed-text

# 4. Start
python run_graph_3.py
```

---

## 11. Key Q&A for Interview

**Q: Why LangGraph instead of a simple LangChain chain?**
The workflow has parallel branches (Jira/Confluence/Perforce fetch simultaneously via `Send()`), conditional loops (retry if verification fails), and stateful resumption (`InMemorySaver` checkpoint). A linear chain cannot express any of this.

**Q: Why local LLMs / Ollama?**
Enterprise data (Jira tickets, internal code, design docs) is confidential. Sending it to cloud AI APIs would be a security risk. Running locally keeps everything on-premise.

**Q: What is HyDE?**
Hypothetical Document Embeddings. Instead of embedding the question and searching, the LLM first writes a hypothetical answer, then you embed that answer and search the vector store. This bridges the vocabulary gap between how questions are phrased and how answers are stored.

**Q: What is RAG-Fusion / RRF?**
Reciprocal Rank Fusion. When you have results from 5+ retrieval strategies you need to merge ranked lists. Formula: `score = Σ 1 / (rank_in_list + k)`. Documents appearing near the top of multiple lists get the highest final score.

**Q: Why BM25 + FAISS instead of just FAISS?**
FAISS finds documents with similar meaning but misses exact technical identifiers like `MTV-1234` or specific error codes. BM25 catches exact keyword matches. Combined they cover the full spectrum.

**Q: What is the Singleton pattern in Configuration?**
`Configuration.__new__` ensures only one Python instance ever exists. All graph nodes share the same config object without re-reading environment variables on every call.

**Q: How does parallel fetching work in LangGraph?**
Using `Send()` from `langgraph.constants`. This triggers multiple nodes to execute concurrently. The three `fetch_*` nodes run in parallel and LangGraph waits for all three before proceeding to `summarize_data`.

**Q: What is Step-Back Prompting?**
A retrieval technique where the LLM first generates a more abstract/general version of the specific question. This finds background documents that provide context the specific query alone would miss.

---

## 12. Quick Numbers

| Metric | Value |
|---|---|
| Total modules | ~63 |
| Subdirectories under `src/assistant/` | 8 |
| Main graph nodes | ~15 |
| Retrieval techniques in HybridRetriever | 7 |
| Verification layers | 5 |
| Enterprise data sources | 3 (Jira, Confluence, Perforce) |
| LangGraph sub-graphs | 2 (main + fact verification) |
| LLM prompt templates | 7 |
