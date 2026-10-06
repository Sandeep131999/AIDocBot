# Enterprise Agentic RAG — Backend

**v2.0.0** · Python 3.12 · LangGraph · FastAPI · PostgreSQL + pgvector · MCP

A production-style **agentic RAG** backend: a LangGraph corrective-RAG (CRAG) agent with hybrid retrieval, multi-provider LLM routing, guardrails mapped to the **OWASP Top 10 for LLM Applications (2025)**, evaluation, and full observability.

It runs locally with Docker Compose and deploys to **Render's free tier** for demos, so no AWS account is needed. See [Deploy on Render](#deploy-on-render-free-tier).

---

## Deployment Profiles

| Profile | Runs on | Database | Suitable for |
|---|---|---|---|
| **Local** | Docker Compose | pgvector container | Development, tests |
| **Demo** | Render free web service | Neon free (persistent) *or* Render free Postgres (30 days) | Portfolio, interviews, 10–100 user demos; cold starts acceptable |
| **Production** | Render paid plan or AWS ECS/Fargate | Managed Postgres with pgvector | Always-on, backups, SLAs |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│  POST /api/chat   POST /api/chat/stream                          │
│  POST /api/documents/upload   GET /api/documents                │
│  POST /api/evaluate   GET /metrics   GET /api/health            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │    Exact-query TTL Cache  │  Bounded, in-process
          │   (skip graph on a hit)   │
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
          │  Postgres FTS (sparse)    │
          │  + pgvector (dense)       │
          │  Weighted RRF + Rerank    │
          │  HyDE (optional)          │
          └───────────────────────────┘
```

---

## Feature Matrix

| Domain | What's Implemented |
|---|---|
| **Core Engineering** | async/await throughout, Pydantic v2, typed state, decorators, FastAPI DI, background tasks |
| **LLM Fundamentals** | Multi-provider fallback (Gemini→Groq→OpenRouter→Anthropic→OpenAI→Ollama), structured outputs, token cost tracking |
| **RAG** | Hybrid PostgreSQL full-text + pgvector search, weighted RRF fusion, cross-encoder reranking, HyDE, CRAG, metadata filtering |
| **Agent Frameworks** | LangGraph CRAG graph, ReAct, reflection/self-correction, supervisor/multi-agent, PostgreSQL checkpointer (durable, resumable runs), human-in-the-loop |
| **MCP** | FastMCP server over Streamable HTTP: tools (`search_knowledge_base`, `duckduckgo_search`, `index_document`, `ask_agent`), resources, prompts |
| **Evaluation** | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall), Hit@K, Precision@K, Recall@K, MRR, NDCG@K, golden datasets |
| **Observability** | LangSmith tracing, OTEL spans, Prometheus metrics (latency, tokens, cost, cache hits, guard blocks), structlog JSON |
| **Guardrails** | 5-layer input guard (length→injection regex→toxic→Presidio PII→LLM), output hallucination scoring, mapped to [OWASP LLM Top 10 2025](#guardrails-owasp-llm-top-10-2025) |
| **Deployment** | Multi-stage Dockerfile, non-root user, health checks, docker-compose (app+postgres+mcp), Render Blueprint (`render.yaml`) |

---

## 2026 Stack Alignment

Where this project sits against the current agentic-AI stack, including what is **not** done yet.

| Area | 2026 expectation | Status |
|---|---|---|
| Agent runtime | Durable, resumable graph execution with human-in-the-loop | **Implemented** — LangGraph + Postgres checkpointer |
| Retrieval | Hybrid (sparse + dense) with reranking, query rewriting, corrective loops | **Implemented** |
| Tool protocol | MCP server over Streamable HTTP | **Implemented** |
| MCP authorization | OAuth 2.1 resource-server auth with protected-resource metadata; stateless requests (spec revision 2026-07-28) | Roadmap |
| Agent-to-agent | A2A v1.0 (Agent Card, task exchange) for cross-agent calls | Roadmap |
| Evaluation | Offline metrics **and** a regression gate in CI | **Partial** — metrics implemented, CI gate on roadmap |
| Observability | Traces, metrics, cost; OpenTelemetry GenAI semantic conventions | **Partial** — OTEL/Prometheus/LangSmith in place; adopt `gen_ai.*` attribute names |
| Security | OWASP LLM Top 10 (2025) mapping, PII controls, rate limits | **Implemented** with known gaps (see table) |
| Cost & latency | Multi-provider routing, caching, cost tracking | **Implemented** — exact-match cache; semantic cache on roadmap |
| Memory | Short-term trimming + persistent long-term memory | **Implemented** |

---

## Quick Start (Local)

### 1. Clone and configure

```bash
cp .envDummy .env
# PostgreSQL is required for pgvector and persistent agent checkpoints.
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
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run with Docker Compose (full stack)

```bash
docker compose up -d                 # app + PostgreSQL/pgvector
docker compose --profile mcp up -d   # also start MCP server
docker compose --profile dev up -d   # also start Adminer (Postgres UI)
```

---

## Deploy on Render (Free Tier)

Render's free tier can host this backend for demos. Read the limits first: they change how the app must be configured.

### What the free tier gives you

Verified against Render's docs (September 2026):

| Resource | Free-tier reality |
|---|---|
| Web service compute | 512 MB RAM, 0.1 CPU |
| Idle behavior | Spins down after 15 min without traffic; ~1 min to wake on the next request |
| Filesystem | Ephemeral: lost on redeploy, restart **and spin-down**; no persistent disk |
| Instance hours | 750 free hours per workspace per month (spun-down time isn't counted) |
| Free Postgres | 1 GB, **expires 30 days after creation** (14-day grace, then deleted), no backups, one free DB per workspace |
| Networking | Free web services cannot *receive* private-network traffic |
| Restarts | Render may restart free services at any time |

### Recommended architecture

```
 Browser / Frontend (Render static site, Vercel, ...)
          │  HTTPS
          ▼
 ┌──────────────────────────────┐        ┌───────────────────────────┐
 │ Render Web Service (free)    │ ─────► │ Postgres + pgvector       │
 │ FastAPI + LangGraph          │        │ Neon free (persistent)    │
 │ region: singapore            │        │  or Render free (30 days) │
 └──────────────┬───────────────┘        └───────────────────────────┘
                │ HTTPS
                ▼
   LLM providers (Gemini / Groq / OpenRouter, free tiers via fallback chain)
```

**Database choice**

- **Neon free** is the better fit for a demo you want to keep up: pgvector is supported, the free plan is permanent, and compute scales to zero after 5 minutes idle. Check Neon's current free-tier storage quota before loading large corpora. Pick a Neon region near your Render region.
- **Render free Postgres** is fine for a short-lived demo (pgvector is supported: `CREATE EXTENSION vector;`), but plan for the 30-day expiry.

### Memory budget: the full stack does not fit in 512 MB

The default local profile loads PyTorch, a sentence-transformers embedder, a cross-encoder reranker and Presidio's NER model in one process. That combination will likely exceed 512 MB and get the service OOM-killed. On 0.1 CPU the cross-encoder is also slow. Use a lighter profile on Render:

| Component | Local (full) | Render free (lite) |
|---|---|---|
| Embeddings (`BAAI/bge-small-en-v1.5`, 384-dim) | sentence-transformers / PyTorch | Same model via an ONNX runtime such as `fastembed`†: same 384 dims, so no re-index, much lower RAM |
| Cross-encoder reranking | On | Off (`RERANKER_ENABLED=false`†); rely on RRF fusion |
| PII detection | Presidio + large spaCy model | Small spaCy model or regex-only mode† |
| HyDE | Optional | Off by default (one extra LLM call per query) |
| Prometheus / OTEL | On | Prometheus on, OTEL off unless you have a collector |
| Local models (Ollama) | Optional | Not available; use hosted providers |

† These toggles may not exist in `config.py` yet. Add them before relying on the lite profile.

**Bake models into the image.** The filesystem is wiped on every spin-down, so a runtime model download would repeat on every cold start. Download the embedding model during `docker build`.

**Bind to Render's port.** The container must listen on `0.0.0.0:$PORT`:

```dockerfile
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### `render.yaml` (Blueprint)

```yaml
services:
  - type: web
    name: agentic-rag-api
    runtime: docker
    plan: free
    region: singapore              # closest Render region to India
    dockerfilePath: ./Dockerfile
    dockerContext: .
    healthCheckPath: /api/health
    autoDeployTrigger: commit
    envVars:
      - key: APP_ENV
        value: production
      - key: AUTH_ENABLED
        value: "true"
      - key: RATE_LIMIT_ENABLED
        value: "true"
      - key: GUARDRAIL_ENABLED
        value: "true"
      - key: POSTGRES_ENABLED
        value: "true"
      - key: CHECKPOINTER_BACKEND
        value: postgres
      - key: JWT_SECRET_KEY
        generateValue: true
      - key: ADMIN_API_KEY
        generateValue: true
      - key: CORS_ORIGINS
        sync: false                # your frontend URL(s)
      - key: POSTGRES_URL          # Option A: Neon connection string (add sslmode=require)
        sync: false
      # Option B: Render's free Postgres instead of Neon
      # - key: POSTGRES_URL
      #   fromDatabase:
      #     name: rag-db
      #     property: connectionString
      - key: GEMINI_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: OPENROUTER_API_KEY
        sync: false

# Option B only
# databases:
#   - name: rag-db
#     plan: free
#     region: singapore
#     databaseName: rag
#     user: rag
```

`sync: false` prompts for the value once, when the Blueprint is first created. Add new secrets later in the dashboard.

### Deploy steps

1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, select the repo, and fill in the prompted secrets.
3. Enable pgvector once against your database: `psql "$POSTGRES_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"`
4. Wait for the deploy, then check `https://<service>.onrender.com/api/health`.
5. Smoke test:

```bash
BASE=https://<service>.onrender.com
curl -s $BASE/api/health
curl -s -X POST $BASE/api/documents/upload -H "Authorization: Bearer $TOKEN" -F "file=@sample.pdf"
curl -s -X POST $BASE/api/chat -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query":"What does the document say about X?"}'
```

### Demo behavior to plan for

- **Cold starts:** the first request after 15 idle minutes takes ~1 min (plus a database wake-up on Neon). Have the frontend call `GET /api/health` on load and show a "waking the server" state.
- **Keeping warm:** an external pinger every ~14 min works but is unofficial. One always-on free service uses about 720–744 of the 750 monthly hours; a second free service would exhaust the quota.
- **MCP server:** the compose `mcp` profile is a separate container, and free Render services can't receive private-network traffic. Either serve MCP from the same FastAPI app (one service) or run it as a second **public** service.
- **LLM rate limits:** free provider tiers throttle quickly. The fallback chain, exact-match cache and a modest `top_k` are what keep a 10–100 user demo working.
- **Retrieval speed:** make sure the embedding column has an HNSW index (`vector_cosine_ops`) and that `top_k` and rerank depth are small.

### If 512 MB is still too tight

Move the web service to a paid Render compute plan. A `1c-2g` plan (1 CPU, 2 GB) fits the full profile; check Render's pricing page for current rates.

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
  "retrieval_strategy": "hybrid",   // optional: dense | bm25 | hybrid | hyde ("bm25" = Postgres full-text)
  "use_cache": true                  // optional — check the exact-query cache first
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

> **Development only.** This endpoint issues a token for any `user_id` and `role`. Do not expose it on a public deployment (see the [production checklist](#production-checklist)).

---

## Graph Nodes

| Node | Purpose | LLM Used |
|---|---|---|
| `input_guard` | Injection, PII, toxic content (OWASP LLM01/LLM02) | fast LLM (Groq) + Presidio |
| `generate_query_or_respond` | ReAct — decide to retrieve or answer directly | default LLM with tools |
| `retrieve` | LangGraph `ToolNode` — runs hybrid search | none (vector DB) |
| `grade_documents` | Structured relevance scoring per doc | fast LLM, structured output |
| `rewrite_question` | Query rewriting for better retrieval | fast LLM |
| `generate_answer` | Final synthesis from graded context | quality LLM |
| `reflect` | Self-correction — check for hallucinations/gaps | fast LLM, structured output |
| `output_guard` | PII scan, hallucination score, prompt-leak check on output | guardrail LLM |

---

## Retrieval Pipeline

```
Query
  │
  ├─→ [HyDE] Generate hypothetical document → embed hypothetical doc
  │
  ├─→ Dense:  pgvector cosine similarity → filter by MIN_RELEVANCE_SCORE
  │
  ├─→ Sparse: PostgreSQL full-text ranking with a GIN index
  │
  └─→ Weighted RRF fusion (0.6 dense / 0.4 sparse)
        │
        └─→ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top-K
```

The sparse leg uses PostgreSQL's built-in full-text ranking. It is a lexical signal like BM25 but not BM25 itself.

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

Each chain uses LangChain's native `.with_fallbacks()` — automatic failover with circuit breaker tracking. On Render, drop `Ollama` from the chains: there is no local model server.

---

## Guardrails (OWASP LLM Top 10, 2025)

**Input guard — 5 layers:**
1. Length check (instant, zero cost)
2. Regex injection patterns (instant)
3. Regex toxic content (instant)
4. Presidio NER + regex PII detection
5. LLM structured check for ambiguous queries

**Output guard:**
- Presidio PII scan on the generated answer
- LLM hallucination score (0–1) checked against `GUARDRAIL_HALLUCINATION_THRESHOLD`
- System prompt leakage detection

**Mapping to the 2025 list:**

| ID | Risk | Control in this repo | Known gap |
|---|---|---|---|
| LLM01 | Prompt Injection | Input guard layers 1, 2, 5 | Indirect injection via ingested documents |
| LLM02 | Sensitive Information Disclosure | Presidio + regex PII on input and output | — |
| LLM03 | Supply Chain | Pinned lockfile, multi-stage non-root image | Dependency/image scanning in CI |
| LLM04 | Data & Model Poisoning | Authenticated, RBAC-gated ingestion | Content scanning of uploads |
| LLM05 | Improper Output Handling | Pydantic structured outputs, output guard | — |
| LLM06 | Excessive Agency | Human-in-the-loop checkpoints, RBAC | Per-tool least-privilege scopes for MCP |
| LLM07 | System Prompt Leakage | Output-guard leakage detection | — |
| LLM08 | Vector & Embedding Weaknesses | Metadata filtering | Per-tenant isolation (e.g. row-level security) |
| LLM09 | Misinformation | Reflection node, hallucination score, cited sources | — |
| LLM10 | Unbounded Consumption | Rate limiting, token/cost tracking, bounded cache, `top_k` | Per-user token budgets |

---

## Evaluation

`POST /api/evaluate` runs RAGAS (faithfulness, answer relevancy, context precision, context recall) and classical retrieval metrics (Hit@K, Precision@K, Recall@K, MRR, NDCG@K) against a golden dataset. Keep the golden set in version control so results stay comparable across changes. Wiring it into CI as a regression gate is on the [roadmap](#roadmap).

---

## Observability

| Tool | Config Key | What it tracks |
|---|---|---|
| LangSmith | `LANGSMITH_ENABLED=true` | Full run traces, token costs, eval datasets |
| Prometheus | `PROMETHEUS_ENABLED=true` | `GET /metrics` — latency, tokens, cost, cache hits, guard blocks |
| OpenTelemetry | `OTEL_ENABLED=true` | Distributed traces via OTLP → Jaeger/Tempo/etc. |
| structlog | Always on | JSON logs in production, pretty-print in development |

On Render, logs go to the dashboard (and optional log streams). Keep `/metrics` behind auth or off the public internet.

---

## Configuration

All config lives in `.env`. See `.envDummy` for the full annotated reference.

Key groups: `App`, `Chunking`, `Embedding`, `VectorStore`, `Retrieval`, `LLM Providers`, `Auth`, `RateLimit`, `PostgreSQL`, `LangSmith`, `OTEL`, `Prometheus`, `MCP`, `Guardrails`, `Prompts`.

Embeddings use the free local `BAAI/bge-small-en-v1.5` model (384 dimensions), cached once per process. Locally, Docker persists the model cache across restarts. On Render the disk is ephemeral, so download the model at image-build time. DuckDuckGo search is free and does not require an API key.

Existing ChromaDB data is not migrated automatically. Re-upload or re-index those documents after starting PostgreSQL with pgvector enabled.

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
python -m src.integrations.mcp_server

# Or via Docker Compose
docker compose --profile mcp up -d
```

Connect at `http://localhost:8001` (streamable-HTTP transport).

**Tools:** `search_knowledge_base`, `duckduckgo_search`, `index_document`, `list_indexed_documents`, `ask_agent`
**Resources:** `rag://status`, `rag://config`
**Prompts:** `rag_system_prompt`, `document_qa_prompt`

Before exposing this server publicly, add authentication (see the roadmap item on MCP OAuth). An open MCP endpoint can call `index_document` and `ask_agent`.

---

## Project Structure

```
backend/
├── api/
│   ├── auth.py          # JWT + API Key auth, RBAC, audit logging
│   ├── main.py          # FastAPI app — JSON and SSE endpoints
│   └── middleware.py    # RequestID, AuditLog, PII scrub, security headers
├── src/
│   ├── cache.py         # Bounded in-process exact-query TTL cache
│   ├── config.py        # pydantic-settings — all env vars typed and validated
│   ├── document_loader.py  # Multi-format async loader (PDF/DOCX/PPTX/XLSX/...)
│   ├── embeddings.py    # Local BGE-small embeddings, @lru_cache
│   ├── evaluator.py     # Classical metrics (Hit@K, MRR, NDCG) + RAGAS
│   ├── graph.py         # LangGraph StateGraph — full CRAG + supervisor
│   ├── guardrails.py    # OWASP LLM guardrails, Presidio PII, hallucination check
│   ├── mcp_server.py    # FastMCP server — tools, resources, prompts
│   ├── memory.py        # Short-term (trim) + long-term PostgreSQL memory
│   ├── multi_llm.py     # Multi-provider fallback, circuit breaker, cost table
│   ├── nodes.py         # All 8 async LangGraph node functions + routing
│   ├── observability.py # LangSmith, OTEL, Prometheus, structlog
│   ├── prompts.py       # PromptLoader + Pydantic structured output schemas
│   ├── state.py         # AgentState — all graph state fields typed
│   ├── tools.py         # LangChain tools with clear schemas
│   └── vector_store.py  # PostgreSQL full-text + pgvector, RRF, reranking, HyDE
├── tests/
│   ├── conftest.py      # Shared fixtures
│   ├── test_api.py      # FastAPI endpoint tests
│   ├── test_config.py   # Config validation tests
│   ├── test_guardrails.py  # OWASP guardrail tests
│   ├── test_nodes.py    # LangGraph node unit tests
│   └── test_retrieval.py   # Full-text, RRF, evaluator metric tests
├── Dockerfile           # Multi-stage build, Python 3.12, non-root user
├── docker-compose.yml   # App + PostgreSQL/pgvector + MCP + Adminer
├── render.yaml          # Render Blueprint (free-tier demo profile)
├── pyproject.toml       # All dependencies with version pins
├── requirements.txt     # Pinned lockfile (generated by uv export)
└── .envDummy            # Fully annotated config template
```

---

## Production Checklist

- [ ] `APP_ENV=production`
- [ ] `AUTH_ENABLED=true` with a strong `JWT_SECRET_KEY`
- [ ] `ADMIN_API_KEY` set to a random 32-byte value
- [ ] **`/auth/token` disabled, or restricted to admins, in production**
- [ ] `CORS_ORIGINS` set to your frontend domain(s)
- [ ] `/metrics` and `/api/evaluate` not publicly reachable (they cost money or leak internals)
- [ ] `POSTGRES_ENABLED=true` and `CHECKPOINTER_BACKEND=postgres`
- [ ] `GUARDRAIL_ENABLED=true`, `PII_DETECTION_ENABLED=true`
- [ ] `RATE_LIMIT_ENABLED=true`
- [ ] Spend caps or quotas set on every LLM provider key
- [ ] `LANGSMITH_ENABLED=true` for production tracing
- [ ] Database backups enabled (not available on Render's free Postgres)
- [ ] If using a pooled Postgres endpoint (e.g. PgBouncer transaction mode), disable prepared statements in the driver or use the direct endpoint for the LangGraph checkpointer

---

## Optional: AWS (ECS/Fargate)

For an always-on production deployment:

1. Build and push to ECR: `docker build -t rag-backend . && docker push <ecr-uri>`
2. Store secrets in Secrets Manager and reference them from the ECS task definition
3. Use RDS PostgreSQL with the pgvector extension for `POSTGRES_URL`
4. CloudWatch for container logs; export OTEL to X-Ray or Grafana Cloud
5. ALB → ECS service for public exposure (add WAF and CloudFront as needed)

---

## Roadmap

- [ ] **MCP authorization:** OAuth 2.1 resource-server auth with protected-resource metadata; align with the stateless 2026-07-28 spec revision
- [ ] **A2A v1.0:** publish an Agent Card and accept tasks from other agents
- [ ] **Eval gate in CI:** fail the build when RAGAS or retrieval metrics regress against the golden set
- [ ] **OTel GenAI conventions:** adopt `gen_ai.*` attributes for spans and metrics
- [ ] **Semantic cache:** embedding-similarity cache backed by pgvector, alongside the exact-match cache
- [ ] **Indirect prompt-injection defense:** sanitize and isolate retrieved content
- [ ] **Multi-tenancy:** per-tenant isolation with Postgres row-level security
- [ ] **True BM25:** replace full-text ranking on the sparse leg if benchmarks justify it
- [ ] **OWASP Top 10 for Agentic Applications:** map agent and MCP controls as a second threat model

---

## License

MIT