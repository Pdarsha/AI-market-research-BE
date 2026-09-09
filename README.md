# CapitalEdge AI Enterprise — Backend

Production-grade asynchronous AI backend for conversational intelligence, hybrid OCR document ingestion, vector semantic search, Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP) financial connectors, dynamic skill injection, and autonomous agent workflow orchestration.

---

## Key Features

* **High-Performance Asynchronous API:** Built with **FastAPI** on Python 3.10+ (tested on Python 3.14) with Uvicorn/Gunicorn.
* **Vector Semantic Search:** **PostgreSQL + `pgvector`** with 1536-dimensional OpenAI vector embeddings and HNSW cosine distance indexing (`ix_document_chunks_embedding`).
* **Two-Stage Intelligent RAG:** First-stage pgvector cosine similarity search followed by second-stage LLM cross-encoder listwise re-ranking (`RerankService`), with automatic web search fallback (DEF-011, DEF-012).
* **Multi-Modal Document Ingestion:** Hybrid extraction pipeline supporting PDF, DOCX, XLSX, XLS, CSV, TXT, MD, and image OCR (PNG, JPG, TIFF, BMP, WEBP) using PyMuPDF native extraction and OpenCV + RapidOCR / Tesseract in parallel thread pools.
* **Model Context Protocol (MCP) Connectors:** Integrated tools for SEC EDGAR (10-K, 10-Q facts), Roic.ai (financial ratios and transcripts), UK Companies House, and Alpha Vantage (market news and sentiment), plus a standalone FastMCP server.
* **Autonomous Agent Workflow Engine:** Trigger-agnostic multi-step agent execution engine with dynamic LLM planning, skill resolution, tool execution, exponential backoff retries, shared blackboard state (`AgentSharedState`), and step logs.
* **Background Scheduler:** Lifespan-managed polling scheduler evaluating cron expressions and intervals every 30 seconds, enforcing execution quotas, recovering stale jobs on reboot, and sending HTML summary emails via SMTP.
* **Document Takeaways Matrix Grid:** Asynchronous comparative analysis grids extracting structured insights across multiple workspace documents with CSV, JSON, and Markdown export.
* **Decoupled Generation Runtime:** Robust SSE token streaming with pub/sub architecture supporting client reconnection and explicit graceful cancellation via the Stop endpoint.
* **Role-Based Access & Credit Ledger:** Daily credit quotas per subscription tier (`free`, `plus`, `pro`, `enterprise`), row-level locked balance deductions, automatic 00:00 UTC resets, and immutable transaction ledger logging.

---

## Technology Stack

* **Framework:** FastAPI 0.115+
* **Language / Runtime:** Python 3.10+ (Docker: `python:3.10-slim`, tested on Python 3.14)
* **ASGI Server:** Uvicorn / Gunicorn
* **ORM & Database:** SQLAlchemy 2.0+ with PostgreSQL 15+ and `pgvector`
* **Schema Migrations:** Alembic (21 versions: Vv1 to Vv21)
* **LLM & Embeddings:** OpenAI API / OpenRouter (`text-embedding-3-small`, `gpt-4.1-mini`, `gpt-5.6-terra`)
* **Document & OCR:** PyMuPDF (`fitz`), `python-docx`, `openpyxl`, `pandas`, `opencv-python-headless`, `rapidocr-onnxruntime`, `pytesseract`
* **Report Generation:** ReportLab
* **Authentication:** Stateless JWT (`python-jose`), Bcrypt (`passlib`), Google OAuth 2.0 (`google-auth`)
* **Email:** FastAPI-Mail (SMTP)
* **MCP Server:** FastMCP
* **Testing:** Pytest, pytest-cov, pytest-asyncio, Typeguard

---

## Prerequisites

Before setting up the backend, ensure you have the following installed:

1. **Python 3.10+** (Python 3.11, 3.12, 3.13, or 3.14 supported)
2. **PostgreSQL 15+** with the **`pgvector`** extension installed
3. **OpenAI API Key** (or OpenRouter API Key)
4. **Git**
5. *(Optional)* **Tesseract OCR** (only required if `OCR_ENGINE=tesseract`; default is `rapidocr` which requires no OS binaries)

---

## Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Pdarsha/agentic-ai.git
cd agentic-ai
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy or create your `.env` file in the project root:
```bash
cp .env.example .env  # Or create .env directly
```

Configure the essential variables in `.env`:
```env
PROJECT_NAME="CapitalEdge AI Enterprise"
API_V1_STR="/v1"

# Database Connection (PostgreSQL with pgvector)
SQLALCHEMY_DATABASE_URI="postgresql://postgres:password@localhost:5432/agentic_db"

# JWT Secret Key (Generate a secure random string)
AUTH_TOKEN="your-super-secret-jwt-key-min-32-chars"
ACCESS_TOKEN_EXPIRE_MINUTES=20160
REFRESH_TOKEN_EXPIRE_MINUTES=40320

# OpenAI / OpenRouter Configuration
OPENAI_API_KEY="your-openai-or-openrouter-api-key"
OPENAI_BASE_URL="https://openrouter.ai/api/v1"
LLM_MODEL="openai/gpt-4.1-mini"
THINK_LONGER_MODEL="openai/gpt-5.6-terra"
DEEP_RESEARCH_MODEL="openai/gpt-5.6-terra"
EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIMENSIONS=1536
RERANK_MODEL="openai/gpt-4.1-mini"

# OCR Settings (rapidocr uses pure Python ONNX runtime)
OCR_ENABLED=true
OCR_ENGINE="rapidocr"

# External API Integrations (Optional / As needed)
SERPAPI_KEY="your-serpapi-key"
SEC_EDGAR_USER_AGENT="CapitalEdge Research admin@capitaledge.ai"
ROIC_API_KEY="your-roic-api-key"
COMPANIES_HOUSE_API_KEY="your-companies-house-api-key"
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-api-key"

# Mail / SMTP Settings (For OTPs and Scheduled Reports)
MAIL_USERNAME="your-email@gmail.com"
MAIL_PASSWORD="your-app-password"
MAIL_FROM="support@capitaledge.ai"
MAIL_PORT=587
MAIL_SERVER="smtp.gmail.com"
```

---

## Database Setup & Migrations

Ensure PostgreSQL is running and has the `vector` extension enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Run Alembic database migrations:
```bash
# Apply all 21 database migrations
alembic upgrade head
```

---

## Running the Application

### 1. Local Development (Uvicorn with Auto-Reload)
```bash
# Ensure virtual environment is active
source venv/bin/activate

# Run Uvicorn server on port 8000
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```
The API is now live at: `http://localhost:8000`

### 2. Running with Docker Compose
```bash
# Build and start App + PostgreSQL + pgvector
docker compose up -d --build

# View logs
docker compose logs -f app
```

---

## Interactive API Documentation

Once the server is running, explore the interactive documentation:

* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check Endpoint:** [http://localhost:8000/](http://localhost:8000/) returns `{"status": 200, "message": "Server running"}`

---

## Running Tests & Quality Checks

Run the automated Pytest suite with `PYTHONPATH=.`:

```bash
# Run full unit test suite
PYTHONPATH=. pytest

# Run with verbose output
PYTHONPATH=. pytest -v

# Run specific test file
PYTHONPATH=. pytest tests/unit/test_pricing_and_upgrade_plans.py

# Run test coverage report
PYTHONPATH=. pytest --cov=server --cov-report=term-missing
```

---

## Core Architecture Overview

```
HTTP Request / SSE
       │
       ▼
server/main.py (FastAPI Lifespan, CORS, Global Handlers)
       │
       ▼
server/api.py (Central Router -> /v1)
       │
  ┌────┴────────────────────────┬────────────────────────┬────────────────────────┐
  ▼                             ▼                        ▼                        ▼
/v1/user                      /v1/ai                  /v1/workspaces           /v1/agents
Auth, Profiles, Credits      Chat, RAG, SSE          Ingestion, OCR, Chunks   Workflows, Scheduler
  │                             │                        │                        │
  └─────────────────────────────┼────────────────────────┴────────────────────────┘
                                ▼
                       server/services/
             (Business Logic, RAG, MCP, Agents)
                                │
                                ▼
                         server/crud/
                   (SQLAlchemy Data Access)
                                │
                                ▼
                 PostgreSQL 15+ with pgvector
        (Tables: users, workspaces, document_chunks,
         chat_sessions, chat_messages, agent_workflows)
```

---

## Background Services & Concurrency

1. **Agent Scheduler (`AgentSchedulerService`):**
   * Polling loop runs in the background of the FastAPI lifespan every 30 seconds.
   * Evaluates cron expressions and frequency schedules (`daily`, `weekly`, `monthly`, `hourly`, `minutes`).
   * Enforces in-memory concurrency locks (`asyncio.Lock`) and execution quotas (`scheduled_count_limit`).
   * Automatically marks orphaned runs as failed on startup via `recover_stale_executions()`.
   * Sends rendered HTML summary emails to recipient emails via SMTP.

2. **In-Flight Generation Runtime (`generation_runtime.py`):**
   * Decouples LLM generation from client HTTP connections via pub/sub queues.
   * Allows browser clients to disconnect/refresh and reconnect without aborting the LLM or losing tokens.
   * Supports immediate, graceful stopping via `POST /v1/ai/chats/{id}/stop` while persisting all tokens generated up to that point.

---

## Production Deployment Runbook

### 1. Automated Linux / EC2 Setup
Run the automated setup script on a fresh Ubuntu 22.04+ or Debian 12 server:
```bash
bash serversetup.sh
```

### 2. Nginx Reverse Proxy Configuration
Configure Nginx with SSL and disabled SSE buffering:
```nginx
server {
    listen 80;
    server_name api.capitaledge.ai;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.capitaledge.ai;

    ssl_certificate /etc/letsencrypt/live/api.capitaledge.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.capitaledge.ai/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Critical for Server-Sent Events (SSE) token streaming:
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## Troubleshooting Guide

| Issue | Cause | Solution |
|---|---|---|
| `type "vector" does not exist` | `pgvector` extension missing in PostgreSQL | Connect to PostgreSQL as superuser and execute: `CREATE EXTENSION IF NOT EXISTS vector;` then re-run `alembic upgrade head`. |
| `ModuleNotFoundError: No module named 'fastapi'` | Virtual environment not active or dependencies missing | Run `source venv/bin/activate && pip install -r requirements.txt`. Ensure `PYTHONPATH=.` is set when running pytest or scripts. |
| `TesseractNotFoundError` | System Tesseract binary missing | Set `OCR_ENGINE=rapidocr` in `.env` (pure Python ONNX, no OS binary needed) or install Tesseract: `sudo apt-get install -y tesseract-ocr`. |
| SSE Stream drops midway in production | Nginx buffering proxy responses | Add `proxy_buffering off; proxy_read_timeout 300s;` to the Nginx configuration. |
| OpenAI 429 Rate Limit | API rate limit reached | The backend automatically applies exponential backoff retries. Verify API quota or use an OpenRouter endpoint. |
| Scheduled Agents not firing | Scheduler paused or timezone mismatch | Check `GET /v1/agents/scheduler/status`. Verify agent's `schedule_timezone`. Manually trigger evaluation with `POST /v1/agents/scheduler/tick`. |

---

## Documentation Reference

For comprehensive technical architecture, database schemas, and end-to-end knowledge transfer documentation, see:
* **[BACKEND_KT_DOCUMENT.md](BACKEND_KT_DOCUMENT.md)** — Comprehensive Production Backend Knowledge Transfer Document
