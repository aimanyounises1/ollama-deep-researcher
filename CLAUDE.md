# CLAUDE.md - AI Assistant Development Guide

## Project Overview

**Ollama Deep Researcher** is a sophisticated research application that combines web search, enterprise data integration, and LLM-powered analysis to perform comprehensive research on various topics.

### Core Purpose
- Perform deep research using Ollama models with integrated web search
- Extract and analyze data from enterprise sources (JIRA, Confluence, Perforce)
- Generate knowledge graphs and track citations
- Provide real-time progress tracking through a modern web interface

### Key Features
- **Deep Web Research**: Multi-iteration research with integrated search capabilities
- **Knowledge Graph Generation**: Automatically build relationship graphs from findings
- **Citation Tracking**: Manage sources with proper attribution
- **RAG Support**: Advanced retrieval-augmented generation with semantic deduplication
- **Vector Store**: FAISS-based vector storage for efficient similarity search
- **LangGraph Integration**: State machine-based research workflows

---

## Architecture & Components

### Overall Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Frontend      │ ───> │   Flask API      │ ───> │   Ollama LLM    │
│   (Next.js)     │      │   (Python)       │      │   (Local)       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                                  ├──> Tavily/Perplexity (Web Search)
                                  ├──> JIRA API
                                  ├──> Confluence API
                                  ├──> Perforce (P4)
                                  └──> FAISS (Vector Store)
```

### Backend Components

#### 1. LangGraph State Machine (`src/assistant/graph_3.py`)
The core orchestration engine managing the research workflow. Main entry point: `create_research_graph()`.

**Key Responsibilities:**
- Coordinate research iterations
- Manage state transitions
- Handle data collection from multiple sources
- Execute verification and analysis pipelines

#### 2. Research State (`src/assistant/state.py`)
Centralized state management using `ResearchState` class.

**State Categories:**
- **Core State**: research_topic, search_query, running_summary, messages
- **Raw Tool Results**: jira_data, confluence_data, perforce_data, web_data
- **Processed Results**: summaries, analysis results, verification data
- **Control Flow**: loop counters, chunk indices, deduplication tracking

#### 3. Tool Integration (`src/assistant/tools/`)

| Tool | Purpose | File |
|------|---------|------|
| JIRA | Issue tracking data | `jira_tool.py`, `tool_jira.py` |
| Confluence | Documentation search | `SolutionBookTool.py`, `tool_confluence.py` |
| Perforce | Code repository data | `PerforceTool.py`, `tool_perforce.py` |
| Security | Vulnerability analysis | `security_analyzer.py` |

#### 4. Verification System (`src/assistant/verifiers/`)

Anti-hallucination framework with multiple verification layers:
- **Fact Checker**: Cross-reference claims with sources
- **Chain of Knowledge**: Multi-hop reasoning verification
- **Multi-Agent Verification**: Consensus-based validation
- **Self-Verification**: Internal consistency checks
- **Tool Output Verification**: Validate external data quality

#### 5. Research Modules (`src/assistant/research/`)
- `analysis.py`: Research result analysis
- `core.py`: Core research logic
- `engine.py`: Research execution engine
- `search.py`: Search coordination
- `knowledge_graph.py`: Graph generation

#### 6. RAG System (`src/assistant/rag/`)
- Semantic deduplication (threshold: 0.95)
- Source weighting based on relevance and freshness
- Hybrid search (vector + keyword)
- FAISS vector indexing

---

## Codebase Structure

```
ollama-deep-researcher/
├── src/
│   ├── assistant/           # Core research logic
│   │   ├── graph_3.py      # Main LangGraph workflow (60K lines)
│   │   ├── graph_2.py      # Legacy workflow (291K lines)
│   │   ├── state.py        # State management
│   │   ├── cli.py          # Command-line interface
│   │   ├── configuration.py # Config management
│   │   ├── prompts.py      # LLM prompts
│   │   ├── research/       # Research modules
│   │   │   ├── analysis.py
│   │   │   ├── core.py
│   │   │   ├── engine.py
│   │   │   ├── search.py
│   │   │   └── knowledge_graph.py
│   │   ├── rag/            # RAG implementations
│   │   ├── tools/          # Integration tools
│   │   │   ├── jira_tool.py
│   │   │   ├── tool_confluence.py
│   │   │   ├── tool_perforce.py
│   │   │   ├── PerforceTool.py
│   │   │   ├── SolutionBookTool.py
│   │   │   └── security_analyzer.py
│   │   ├── verifiers/      # Anti-hallucination
│   │   │   ├── fact_checker.py
│   │   │   ├── chain_of_knowledge.py
│   │   │   ├── multi_agent_verification.py
│   │   │   ├── self_verification.py
│   │   │   └── tool_output_verification.py
│   │   ├── summarizers/    # Content summarization
│   │   │   ├── map_rerank.py
│   │   │   └── perforce_summarizer.py
│   │   ├── processors/     # Data processors
│   │   ├── analyzers/      # Analysis modules
│   │   ├── memory/         # Memory management
│   │   └── utils/          # Assistant utilities
│   ├── utils/              # Global utilities
│   │   ├── config.py
│   │   ├── prompts.py
│   │   ├── ssl_fix.py
│   │   ├── proxy_helper.py
│   │   └── content_cleaner.py
│   ├── tests/              # Test suite
│   └── components/         # Shared components
├── frontend/               # React frontend
│   └── src/
│       ├── components/     # UI components
│       ├── pages/          # Page components
│       └── services/       # API services
├── vector_store/           # FAISS indices
├── run_graph_3.py         # Main entry point (45K lines)
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── langgraph.json         # LangGraph config
└── .env.example           # Environment template
```

---

## Key Technologies & Dependencies

### Core Technologies
- **Python**: 3.9+ (project uses 3.11)
- **Node.js**: 16+ (for frontend)
- **LangChain**: 0.3.25 (LLM orchestration)
- **LangGraph**: 0.4.3 (state machine workflows)
- **Ollama**: Local LLM inference

### Major Python Dependencies

```python
# LLM & AI
langchain==0.3.25
langchain-community==0.3.23
langchain-core==0.3.59
langchain-ollama==0.3.2
langgraph==0.4.3
sentence-transformers==4.1.0

# Enterprise Integration
jira==3.8.0
P4Python==2024.2.2682690

# Web & HTTP
aiohttp==3.11.18
requests==2.32.3
urllib3==2.4.0
tavily-python==0.7.2

# Data Processing
pandas==2.2.3
beautifulsoup4==4.13.4
html2text==2025.4.15
trafilatura==2.0.0

# Utilities
python-dotenv==1.1.0
rich==14.0.0
typer==0.15.3
tenacity==9.1.2
```

### Frontend Stack
- **React**: UI framework
- **Next.js**: React framework
- **Tailwind CSS**: Styling

---

## Development Workflow

### Running the Application

#### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: TAVILY_API_KEY, PERPLEXITY_API_KEY, OLLAMA_ENDPOINT
```

#### 2. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

#### 3. Start Services

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start backend
python run_graph_3.py --debug "Your research query"

# Terminal 3: Start frontend
cd frontend
npm start
```

#### 4. Access Application
- Frontend: http://localhost:3000
- Backend API: Embedded in Flask

### Command-Line Interface

```bash
# Basic research
python run_graph_3.py "How does quantum computing work?"

# With options
python run_graph_3.py \
  --query "Your question" \
  --debug \
  --max-iterations 150 \
  --output results.json \
  --fast-mode

# Available flags:
# --debug              Enable debug logging
# --no-rag            Disable RAG
# --no-verification   Disable verification
# --no-cross-source   Disable cross-source analysis
# --fast-mode         Quick testing mode
# --max-iterations    Max graph iterations (default: 150)
# --max-runtime       Max runtime in seconds (default: 1800)
# --timeout           Overall timeout
# --output            Output file path
```

### Configuration Options

Via `.env` file or environment variables:

```bash
# Web Search APIs
TAVILY_API_KEY=your_key
PERPLEXITY_API_KEY=your_key

# Ollama Configuration
OLLAMA_ENDPOINT=http://localhost:11434
LLM_MODEL=qwen3:30b-a3b           # Tool-compatible model
EMBEDDING_MODEL=nomic-embed-text:latest

# Research Settings
OLLAMA_MAX_WEB_RESEARCH_LOOPS=3
OLLAMA_INCLUDE_WEB_SEARCH=true
OLLAMA_GENERATE_KNOWLEDGE_GRAPH=true
OLLAMA_TRACK_CITATIONS=true

# RAG Configuration
RAG_SOURCE_WEIGHT=0.6
RAG_CONTENT_WEIGHT=0.4
SEMANTIC_DEDUP_THRESHOLD=0.95
FRESHNESS_DECAY_DAYS=365

# LangSmith (Optional)
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=ollama-deep-researcher
```

---

## Important Patterns & Best Practices

### 1. Model Compatibility

**Critical**: Not all models support tool calling.

```python
# ✅ SUPPORTED (tool-compatible)
LLM_MODEL=qwen3:30b-a3b
LLM_MODEL=llama3:latest
LLM_MODEL=mistral:latest

# ❌ NOT SUPPORTED
LLM_MODEL=deepseek-r1:*  # Will cause 400 errors
```

The `run_graph_3.py` automatically detects and falls back from DeepSeek-R1 models:

```python:run_graph_3.py
_llm_env = os.environ.get("LLM_MODEL", "").lower()
if _llm_env.startswith("deepseek-r1"):
    logging.warning("Detected unsupported DeepSeek-R1 model. Falling back to 'qwen3:30b-a3b'")
    os.environ["LLM_MODEL"] = "qwen3:30b-a3b"
```

### 2. State Management

Always use the `ResearchState` class for type-safe state access:

```python
from src.assistant.state import ResearchState

# Initialize state
state = ResearchState({
    "search_query": query,
    "research_topic": query,
    "running_summary": "",
    "research_loop_count": 0,
    "messages": [HumanMessage(content=query)]
})

# Safe access with defaults
value = state.get_state_value("key", default=None)

# Batch updates
state.update_state({
    "running_summary": new_summary,
    "research_loop_count": state["research_loop_count"] + 1
})
```

### 3. Error Handling & Recovery

The system implements comprehensive error handling:

```python
# Automatic retry with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=16)
)
def fetch_data_with_retry():
    # Network operation
    pass
```

### 4. Memory Management

For long-running research:

```python
import gc

# Force garbage collection after heavy operations
result = process_large_data()
del large_object
gc.collect()
```

### 5. SSL & Proxy Configuration

The application handles corporate proxy environments:

```python:run_graph_3.py
# SSL verification disabled for internal development
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Proxy configuration
os.environ["HTTP_PROXY"] = os.environ.get("HTTP_PROXY", "")
os.environ["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY", "")
```

### 6. Deduplication Tracking

Prevent duplicate data collection:

```python
# State includes deduplication sets
state["_jira_keys_seen"] = set()          # JIRA keys
state["_perforce_cl_numbers"] = set()     # Perforce CLs
state["_confluence_urls_seen"] = set()    # Confluence URLs
state["_content_hashes"] = set()          # Content hashes

# Check before adding
if jira_key not in state["_jira_keys_seen"]:
    state["_jira_keys_seen"].add(jira_key)
    process_jira_issue(issue)
```

### 7. Serialization for JSON Output

Handle LangChain message objects:

```python
def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to JSON-serializable dicts."""
    serialized = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            serialized.append({
                "type": msg.type,
                "content": msg.content,
            })
    return serialized

# Use with json.dump
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2, default=_json_fallback)
```

---

## Testing & Quality

### Running Tests

```bash
# Backend tests
pytest src/tests/

# Frontend tests
cd frontend
npm test
```

### Code Quality Tools

```bash
# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

### Ruff Configuration

From `pyproject.toml`:
- Enabled: pycodestyle (E), pyflakes (F), isort (I), pydocstyle (D)
- Convention: Google-style docstrings
- Ignored: UP006, UP007, UP035, D417, E501

---

## Git Conventions

### Branch Naming
- Feature branches: `feature/feature-name`
- Bug fixes: `fix/bug-description`
- Claude branches: `claude/claude-*-session-id`

### Commit Messages
Follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance

### Force Push Protocol

**IMPORTANT**: See `FORCE_PUSH_MAIN_INSTRUCTIONS.md` for force-push procedures.

Only force-push when:
1. Removing sensitive data
2. Rewriting problematic history
3. User has explicit approval

```bash
# Always create backup first
git branch backup-main main

# Disable branch protection
# Push with --force
git push origin HEAD:main --force

# Verify
git fetch origin main
git log origin/main
```

---

## Security Considerations

### 1. Sensitive Data

**NEVER commit:**
- API keys (use `.env`)
- Corporate JIRA tickets
- Internal URLs
- Perforce changelist details
- Confluence page content
- System architecture diagrams

See `SECURITY_STATUS_REPORT.md` and `SECURITY_CLEANUP_INSTRUCTIONS.md` for details.

### 2. .gitignore Patterns

```gitignore
# Environment
.env
.env.*
!.env.example

# API Keys & Secrets
*.key
*.pem
secrets/

# Output Files
graph_3_result*.json
*.log

# Vector Store
vector_store/

# Corporate Data
examples/
*_enterprise_*
*_corporate_*
```

### 3. Environment Variables

Always use environment variables for:
- API keys (TAVILY_API_KEY, PERPLEXITY_API_KEY)
- Endpoints (OLLAMA_ENDPOINT)
- Database credentials
- Internal service URLs

### 4. SSL Configuration

For production, enable SSL verification:

```python
# Development (current)
ssl._create_default_https_context = ssl._create_unverified_context

# Production (recommended)
# Remove SSL bypass, use proper certificates
```

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

```
Error: Cannot connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check endpoint in .env
OLLAMA_ENDPOINT=http://localhost:11434
```

#### 2. Tool Calling Not Supported

```
Error: 400 - Model does not support tools
```

**Solution:**
```bash
# Change to tool-compatible model
LLM_MODEL=qwen3:30b-a3b
# NOT: deepseek-r1:*
```

#### 3. Max Iterations Exceeded

```
Warning: Research exceeded maximum iterations (150)
```

**Solution:**
```bash
# Increase max iterations
python run_graph_3.py --max-iterations 250 "query"

# Or enable fast mode
python run_graph_3.py --fast-mode "query"
```

#### 4. Memory Issues

```
Error: Out of memory
```

**Solution:**
- Enable fast mode: `--fast-mode`
- Reduce max iterations
- Increase Docker memory limits
- Force garbage collection in code

#### 5. DNS Resolution Errors

```
socket.gaierror: [Errno -2] Name or service not known
```

**Solution:**
- Check proxy configuration
- Verify VPN connection (for corporate services)
- Check NO_PROXY settings

```bash
export NO_PROXY="localhost,127.0.0.1,deljira,internal-domains"
```

#### 6. LangSmith Connection Issues

The application handles LangSmith proxy routing automatically:

```python:run_graph_3.py
# Bypasses proxy for LangSmith when no corporate proxy set
_langsmith_domain = "api.smith.langchain.com"
if not _proxy_vars_present and _langsmith_domain not in _no_proxy:
    os.environ["NO_PROXY"] = f"{_no_proxy},{_langsmith_domain}"
```

---

## Performance Optimization

### 1. Fast Mode

For quick testing, use fast mode:

```bash
python run_graph_3.py --fast-mode "query"
```

Fast mode disables:
- Cross-source analysis
- Map-rerank operations
- Chain of Knowledge verification
- Multi-agent verification

Reduced timeouts:
- LLM: 30s (from 60s)
- API: 60s (from 120s)
- Node: 90s (from 180s)

### 2. Iteration Control

Default: 150 iterations, ~30 minutes max runtime

Adjust based on query complexity:
- Simple queries: `--max-iterations 50`
- Complex research: `--max-iterations 250`
- Quick test: `--fast-mode --max-iterations 20`

### 3. Partial Results

The system auto-saves partial results when:
- Max iterations exceeded
- Timeout reached
- User interruption (Ctrl+C)
- Errors occur

Files saved as: `graph_3_result_{reason}_{timestamp}.json`

---

## Data Flow

### Research Execution Flow

```
1. User Query
   ↓
2. Initialize State (ResearchState)
   ↓
3. LangGraph Workflow (graph_3.py)
   ↓
4. Data Collection Phase
   ├─→ JIRA Tool
   ├─→ Confluence Tool
   ├─→ Perforce Tool
   └─→ Web Search (Tavily/Perplexity)
   ↓
5. RAG Processing
   ├─→ Semantic Deduplication
   ├─→ Vector Storage (FAISS)
   └─→ Hybrid Retrieval
   ↓
6. Analysis & Verification
   ├─→ Fact Checking
   ├─→ Chain of Knowledge
   ├─→ Multi-Agent Verification
   └─→ Cross-Source Analysis
   ↓
7. Summarization
   ├─→ Map-Rerank
   ├─→ Knowledge Graph Generation
   └─→ Citation Tracking
   ↓
8. Output Generation
   └─→ JSON with summaries, citations, graphs
```

### State Evolution

```python
# Initial State
{
    "search_query": "user query",
    "research_loop_count": 0,
    "running_summary": "",
    "jira_data": {},
    "confluence_data": {},
    "perforce_data": {}
}

# After Data Collection
{
    "jira_data": {
        "raw_results": [...],
        "summary": "..."
    },
    "confluence_data": {
        "raw_results": [...],
        "summary": "..."
    },
    "running_summary": "Initial findings..."
}

# Final State
{
    "final_summary": "Complete research summary...",
    "knowledge_graph": {...},
    "citations": [...],
    "verification_results": {...},
    "completed_research": true,
    "_data_counts": {
        "jira_count": 15,
        "perforce_count": 23,
        "confluence_count": 8
    }
}
```

---

## API Integration Details

### JIRA Configuration

Required environment (corporate):
- JIRA endpoint
- Authentication credentials
- Project filters

### Confluence (SolutionBook)

Integration via `SolutionBookTool.py`:
- Document search
- Content extraction
- Attachment processing

### Perforce

P4Python integration:
- Changelist queries
- Diff analysis
- Code review data

### Web Search APIs

**Tavily:**
- General web search
- Configurable result limits

**Perplexity:**
- AI-powered search
- Conversational results

---

## Extending the System

### Adding a New Tool

1. Create tool file in `src/assistant/tools/`:

```python
# src/assistant/tools/tool_newservice.py
from typing import Dict, Any, List

class NewServiceTool:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the new service."""
        # Implementation
        return results
```

2. Register in graph (`graph_3.py`):

```python
def create_research_graph():
    # Add to tool initialization
    new_service_tool = NewServiceTool(config)

    # Add node
    graph.add_node("fetch_newservice", fetch_newservice_node)

    # Add edges
    graph.add_edge("start", "fetch_newservice")
```

3. Update state (`state.py`):

```python
# Add to ResearchState
# newservice_data: Dict[str, Any]
# newservice_summary: str
```

### Adding a New Verifier

1. Create verifier in `src/assistant/verifiers/`:

```python
# src/assistant/verifiers/new_verifier.py
from typing import Dict, Any

class NewVerifier:
    def verify(self, claim: str, context: str) -> Dict[str, Any]:
        """Verify claim against context."""
        return {
            "verified": True,
            "confidence": 0.95,
            "evidence": [...]
        }
```

2. Integrate in graph workflow
3. Add to verification pipeline

---

## LangGraph Specifics

### Graph Configuration

From `langgraph.json`:
```json
{
    "dockerfile_lines": [
        "RUN pip install --no-cache-dir aiohttp urllib3 requests certifi"
    ],
    "graphs": {
        "open_deep_research": "src/assistant/graph_2.py:graph"
    },
    "python_version": "3.11",
    "env": "./.env",
    "dependencies": ["."]
}
```

### State Channels

LangGraph uses "channels" for state management:
- Messages: `Annotated[List[BaseMessage], operator.add]`
- Lists: Append-only with operator.add
- Dicts: Merge updates
- Sets: For deduplication

### Checkpointing

Uses `InMemorySaver` for stateful execution:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = create_research_graph()
result = graph.invoke(state, {
    "configurable": {
        "thread_id": "research_123"
    }
})
```

---

## Frontend Integration

### API Communication

Frontend in `frontend/src/` communicates via:
- REST API for research requests
- WebSocket for real-time progress
- JSON responses with structured data

### Key Frontend Files

```
frontend/src/
├── components/        # React components
├── pages/            # Next.js pages
├── services/         # API clients
└── [other Next.js structure]
```

### Response Format

```json
{
    "final_summary": "Research findings...",
    "jira_data": {
        "summary": "...",
        "raw_results": [...]
    },
    "knowledge_graph": {...},
    "citations": [...],
    "_data_counts": {
        "jira_count": 15,
        "total_count": 46
    }
}
```

---

## Deployment

### Local Development

See "Development Workflow" section above.

### Docker Deployment

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

See `PRODUCTION_DEPLOYMENT.md` for production details.

### Requirements

- Python 3.9+
- Node.js 16+
- Ollama (local or remote)
- 4GB+ RAM recommended
- Docker & Docker Compose (for containerized deployment)

---

## Additional Resources

### Documentation Files

- `README.md` - Main project documentation
- `PRODUCTION_DEPLOYMENT.md` - Production setup
- `NEXTJS_FRONTEND_README.md` - Frontend details
- `SECURITY_STATUS_REPORT.md` - Security audit
- `SECURITY_CLEANUP_INSTRUCTIONS.md` - Data cleanup
- `FORCE_PUSH_MAIN_INSTRUCTIONS.md` - Git force-push guide
- `src/assistant/SearchEnhancementsDocumentation.md` - Search features

### Key Files to Understand

1. **Entry Point**: `run_graph_3.py` (45K lines)
   - Main execution logic
   - Error handling
   - State management
   - Result serialization

2. **Core Graph**: `src/assistant/graph_3.py` (60K lines)
   - LangGraph workflow
   - Node definitions
   - Edge routing
   - Tool integration

3. **State**: `src/assistant/state.py`
   - State schema
   - Helper methods
   - Default values

4. **Configuration**: `src/assistant/configuration.py`
   - Config management
   - Environment handling

---

## Quick Reference

### Environment Variables Cheat Sheet

```bash
# Required
TAVILY_API_KEY=xxx
PERPLEXITY_API_KEY=xxx
OLLAMA_ENDPOINT=http://localhost:11434

# Models
LLM_MODEL=qwen3:30b-a3b
EMBEDDING_MODEL=nomic-embed-text:latest

# Research
OLLAMA_MAX_WEB_RESEARCH_LOOPS=3
OLLAMA_INCLUDE_WEB_SEARCH=true

# Optional
LANGCHAIN_TRACING_V2=false
DEBUG=false
```

### Common Commands

```bash
# Development
python run_graph_3.py --debug "query"

# Production
python run_graph_3.py --query "query" --output results.json

# Fast mode
python run_graph_3.py --fast-mode "query"

# Custom iterations
python run_graph_3.py --max-iterations 200 "query"

# No verification
python run_graph_3.py --no-verification --no-cross-source "query"
```

### File Patterns

```bash
# Research results
graph_3_result*.json

# Logs
run_graph_3.log

# Vector store
vector_store/

# Frontend build
frontend/.next/
frontend/node_modules/
```

---

## Conclusion

This document provides a comprehensive guide for AI assistants working with the Ollama Deep Researcher codebase.

**Key Takeaways:**
1. Always use tool-compatible LLM models (qwen3, llama3, mistral)
2. Manage state through `ResearchState` class
3. Handle corporate proxy/SSL configurations
4. Use deduplication to prevent duplicate data
5. Implement proper error handling and retries
6. Never commit sensitive corporate data
7. Use fast mode for quick testing
8. Monitor iteration counts and timeouts

**For Questions:**
- Check existing documentation in repository root
- Review source code comments
- Examine test files for usage examples
- Consult LangChain/LangGraph official docs

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Maintained By:** AI Assistant (Claude)
