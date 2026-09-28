# Enterprise Agentic RAG — Backend

**v2.0.0** · Python 3.12 · LangGraph · FastAPI · ChromaDB · Redis · PostgreSQL

A production-grade Agentic Retrieval-Augmented Generation system demonstrating enterprise patterns across all 9 domains of the modern AI engineering stack.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│  POST /api/chat   POST /api/chat/stream   WS /api/chat/ws       │
│  POST /api/documents/upload   GET /api/documents                │
│  POST /api/evaluate   GET /metrics   GET /api/health            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │     Semantic Cache        │  Redis exact + ChromaDB semantic
          │  (skip graph if hit)      │  cosine similarity ≥ 0.92
          └─────────────┬─────────────┘
                        │ cache miss
          ┌─────────────▼─────────────────────────────────────────┐
          │              LangGraph StateGraph (CRAG)               │
          │                                                         │
          │  START → input_guard → generate_query_or_respond       │
          │              │               │ tool call               │
          │         [blocked]          retrieve (ToolNode)         │
          │              │               │                         │
          │             END         grade_documents                │
          │                          │         │                   │
          │                    [relevant]   [irrelevant]           │
          │                          │         │                   │
          │                   generate_answer  rewrite_question    │
          │                          │         │                   │
          │                       reflect   [loops back]           │
          │                          │                             │
          │                    output_guard → END                  │
          └─────────────────────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │      Hybrid Retrieval     │
          │  BM25 (sparse) + Dense    │
          │  RRF Fusion + Reranking   │
          │  HyDE (optional)          │
          └───────────────────────────┘
```

---

## Feature Matrix

| Domain | What's Implemented |
|---|---|
| **Core Engineering** | async/await throughout, Pydantic v2, typed state, decorators, FastAPI DI, background tasks |
| **LLM Fundamentals** | Multi-provider fallback (Gemini→Groq→OpenRouter→Anthropic→OpenAI→Ollama), structured outputs, token cost tracking |
| **RAG** | Hybrid BM25+dense, RRF fusion, cross-encoder reranking, HyDE, CRAG, metadata filtering, score thresholding |
| **Agent Frameworks** | LangGraph CRAG graph, ReAct, reflection/self-correction, supervisor/multi-agent, checkpointer (memory/postgres/redis), HITL |
| **MCP** | FastMCP server with tools (`search_knowledge_base`, `index_document`, `ask_agent`), resources, prompts — stdio + HTTP transports |
| **Evaluation** | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall), Hit@K, Precision@K, Recall@K, MRR, NDCG@K, golden datasets |
| **Observability** | LangSmith tracing, OTEL spans, Prometheus metrics (latency, tokens, cost, cache hits, guard blocks), structlog JSON |
| **Guardrails** | 5-layer input guard (length→injection regex→toxic→Presidio PII→LLM), output hallucination scoring, OWASP LLM Top 10 |
| **Deployment** | Multi-stage Dockerfile, non-root user, health checks, docker-compose (app+postgres+redis+mcp), CI-ready |

---

## Quick Start

### 1. Clone and configure

```bash
cp .envDummy .env
# Fill in at least one LLM API key (GEMINI_API_KEY, GROQ_API_KEY, etc.)
# or set OLLAMA_BASE_URL if running locally
```

### 2. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run with Docker Compose (full stack)

```bash
docker compose up -d           # app + postgres + redis
docker compose --profile mcp up -d   # also start MCP server
docker compose --profile dev up -d   # also start Adminer (Postgres UI)
```

---

## API Reference

### Chat

```http
POST /api/chat
Content-Type: application/json

{
  "query": "How do I reset my password?",
  "session_id": "user-123",         // optional — for conversation continuity
  "top_k": 5,                       // optional
  "retrieval_strategy": "hybrid",   // optional: dense | bm25 | hybrid | hyde
  "use_cache": true                  // optional — check semantic cache first
}
```

Response:
```json
{
  "answer": "You can reset your password by...",
  "sources": [{"filename": "faq.pdf", "page": 2, "excerpt": "..."}],
  "session_id": "user-123",
  "guard_blocked": false,
  "cache_hit": false,
  "latency_ms": 842.3,
  "token_usage": {"prompt_tokens": 512, "completion_tokens": 128, "estimated_cost_usd": 0.00004},
  "provider_used": "ChatGoogleGenerativeAI",
  "node_trace": ["input_guard:none", "generate_query_or_respond", "grade_documents:relevant", "generate_answer", "output_guard:safe=True"]
}
```

### SSE Streaming

```http
POST /api/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

Events emitted: `node_start`, `token`, `blocked`, `done`, `error`

### WebSocket

```
WS /api/chat/ws/{session_id}

Send: {"query": "your question"}
Receive: {"event": "token", "data": "..."} × N
         {"event": "done", "answer": "...", "session_id": "..."}
```

### Documents

```http
POST   /api/documents/upload     # multipart/form-data, field: file
GET    /api/documents             # list all indexed files
DELETE /api/documents/{filename}  # remove from vector store
```

### Evaluation

```http
POST /api/evaluate
{
  "questions": ["What is the return policy?"],
  "ground_truths": ["Returns within 30 days"],   // optional
  "relevant_doc_ids": [["policy.pdf"]]            // optional
}
```

### Auth (dev helper)

```http
POST /auth/token
{"user_id": "alice", "role": "user"}
→ {"access_token": "...", "refresh_token": "..."}
```

---

## Graph Nodes

| Node | Purpose | LLM Used |
|---|---|---|
| `input_guard` | OWASP LLM01/06/09 — injection, PII, toxic | fast LLM (Groq) + Presidio |
| `generate_query_or_respond` | ReAct — decide to retrieve or answer directly | default LLM with tools |
| `retrieve` | LangGraph `ToolNode` — runs hybrid search | none (vector DB) |
| `grade_documents` | Structured relevance scoring per doc | fast LLM, structured output |
| `rewrite_question` | Query rewriting for better retrieval | fast LLM |
| `generate_answer` | Final synthesis from graded context | quality LLM |
| `reflect` | Self-correction — check for hallucinations/gaps | fast LLM, structured output |
| `output_guard` | PII scan + hallucination score on output | guardrail LLM |

---

## Retrieval Pipeline

```
Query
  │
  ├─→ [HyDE] Generate hypothetical document → embed hypothetical doc
  │
  ├─→ Dense:  ChromaDB cosine similarity → filter by MIN_RELEVANCE_SCORE
  │
  ├─→ BM25:   Build index from all collection docs → score with Okapi BM25
  │
  └─→ RRF Fusion (0.6 dense + 0.4 BM25)
        │
        └─→ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top-K
```

---

## Multi-LLM Routing

```python
# Provider order: fastest/cheapest → highest quality
get_fast_llm()     # Groq → Gemini Flash → Ollama        (grading, rewriting)
get_llm()          # Gemini → Groq → OpenRouter → Ollama (main agent)
get_quality_llm()  # Gemini → Anthropic → OpenAI → Ollama (answer generation)
get_grader_llm()   # fast + structured output            (document grading)
get_guardrail_llm() # fast + structured output           (security checks)
```

Each chain uses LangChain's native `.with_fallbacks()` — automatic failover with circuit breaker tracking.

---

## Guardrails (OWASP LLM Top 10)

**Input Guard — 5 layers:**
1. Length check (instant, zero cost)
2. Regex injection patterns (instant) — covers LLM01
3. Regex toxic content (instant) — covers LLM06  
4. Presidio NER + regex PII detection — covers LLM06
5. LLM structured check for ambiguous queries — covers LLM01/LLM09

**Output Guard:**
- Presidio PII scan on generated answer
- LLM hallucination score (0–1) checked against `GUARDRAIL_HALLUCINATION_THRESHOLD`
- System prompt leakage detection

---

## Observability

| Tool | Config Key | What it tracks |
|---|---|---|
| LangSmith | `LANGSMITH_ENABLED=true` | Full run traces, token costs, eval datasets |
| Prometheus | `PROMETHEUS_ENABLED=true` | `GET /metrics` — latency, tokens, cost, cache hits, guard blocks |
| OpenTelemetry | `OTEL_ENABLED=true` | Distributed traces via OTLP → Jaeger/Tempo/etc. |
| structlog | Always on | JSON logs in production, pretty-print in development |

---

## Configuration

All config lives in `.env`. See `.envDummy` for the full annotated reference.

Key groups: `App`, `Chunking`, `Embedding`, `VectorStore`, `Retrieval`, `LLM Providers`, `Auth`, `RateLimit`, `Redis`, `PostgreSQL`, `LangSmith`, `OTEL`, `Prometheus`, `MCP`, `Guardrails`, `Prompts`.

Prompts are versioned in `.env` — change them at runtime without redeploying code.

---

## Running Tests

```bash
# Install dev deps
uv sync --extra dev

# Run full suite
uv run pytest

# Run specific suite
uv run pytest tests/test_guardrails.py -v
uv run pytest tests/test_retrieval.py -v
uv run pytest tests/test_nodes.py -v
uv run pytest tests/test_api.py -v

# With coverage
uv run pytest --cov=src --cov-report=html
```

---

## MCP Server

The MCP server exposes the RAG system to any MCP-compatible client (Claude Desktop, Cursor, VS Code with MCP extension).

```bash
# Start standalone
python -m src.mcp_server

# Or via Docker Compose
docker compose --profile mcp up -d
```

Connect at `http://localhost:8001` (streamable-HTTP transport).

**Tools:** `search_knowledge_base`, `index_document`, `list_indexed_documents`, `ask_agent`  
**Resources:** `rag://status`, `rag://config`  
**Prompts:** `rag_system_prompt`, `document_qa_prompt`

---

## Project Structure

```
backend/
├── api/
│   ├── auth.py          # JWT + API Key auth, RBAC, audit logging
│   ├── main.py          # FastAPI app — all endpoints, SSE, WebSocket
│   └── middleware.py    # RequestID, AuditLog, PII scrub, security headers
├── src/
│   ├── cache.py         # Exact (Redis) + semantic (ChromaDB) cache
│   ├── config.py        # pydantic-settings — all env vars typed and validated
│   ├── document_loader.py  # Multi-format async loader (PDF/DOCX/PPTX/XLSX/...)
│   ├── embeddings.py    # HuggingFace or Ollama embeddings, @lru_cache
│   ├── evaluator.py     # Classical metrics (Hit@K, MRR, NDCG) + RAGAS
│   ├── graph.py         # LangGraph StateGraph — full CRAG + supervisor
│   ├── guardrails.py    # OWASP LLM01/06/09, Presidio PII, hallucination check
│   ├── mcp_server.py    # FastMCP server — tools, resources, prompts
│   ├── memory.py        # Short-term (trim) + long-term (Redis) memory
│   ├── multi_llm.py     # Multi-provider fallback, circuit breaker, cost table
│   ├── nodes.py         # All 8 async LangGraph node functions + routing
│   ├── observability.py # LangSmith, OTEL, Prometheus, structlog
│   ├── prompts.py       # PromptLoader + Pydantic structured output schemas
│   ├── state.py         # AgentState — all graph state fields typed
│   ├── tools.py         # LangChain tools with clear schemas
│   └── vector_store.py  # Hybrid BM25+dense, RRF, cross-encoder reranking, HyDE
├── tests/
│   ├── conftest.py      # Shared fixtures
│   ├── test_api.py      # FastAPI endpoint tests
│   ├── test_config.py   # Config validation tests
│   ├── test_guardrails.py  # OWASP guardrail tests
│   ├── test_nodes.py    # LangGraph node unit tests
│   └── test_retrieval.py   # BM25, RRF, evaluator metric tests
├── Dockerfile           # Multi-stage build, Python 3.12, non-root user
├── docker-compose.yml   # App + Postgres + Redis + MCP + Adminer
├── pyproject.toml       # All dependencies with version pins
├── requirements.txt     # Pinned lockfile (generated by uv export)
└── .envDummy            # Fully annotated config template
```

---

## Deployment

### Production Checklist

- [ ] Set `APP_ENV=production`
- [ ] Set `AUTH_ENABLED=true` and strong `JWT_SECRET_KEY`
- [ ] Set `ADMIN_API_KEY` to a random 32-byte hex value
- [ ] Set `CORS_ORIGINS` to your frontend domain(s)
- [ ] Set `REDIS_ENABLED=true` and configure `REDIS_PASSWORD`
- [ ] Set `POSTGRES_ENABLED=true` and `CHECKPOINTER_BACKEND=postgres`
- [ ] Set `LANGSMITH_ENABLED=true` for production tracing
- [ ] Configure `GUARDRAIL_ENABLED=true`, `PII_DETECTION_ENABLED=true`
- [ ] Set `RATE_LIMIT_ENABLED=true`

### AWS Deployment (ECS/Fargate)

1. Build and push to ECR: `docker build -t rag-backend . && docker push <ecr-uri>`
2. Store secrets in Secrets Manager, reference via ECS task definition environment
3. Use RDS Postgres for `POSTGRES_URL`, ElastiCache Redis for `REDIS_URL`
4. CloudWatch for container logs; export OTEL to AWS X-Ray or Grafana Cloud
5. API Gateway → ALB → ECS service for public exposure

---

## License

MIT
