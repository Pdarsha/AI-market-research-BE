# CapitalEdge AI Enterprise — Backend Knowledge Transfer (KT) Document

**Document:** `BACKEND_KT_DOCUMENT.md`  
**System:** CapitalEdge AI Enterprise Backend  
**Version:** 1.0.0  
**Framework:** FastAPI / Python 3.10+ (tested on Python 3.14) / SQLAlchemy / PostgreSQL + `pgvector` / Alembic / OpenAI API / FastMCP  
**Source Code Baseline:** `/mnt/data/projects/agentic-ai`  
**Last Verified:** Production Codebase Inspection (Zero Assumption Baseline)

---

## Table of Contents

1. [Section A: Backend Overview](#section-a-backend-overview)
2. [Section B: Complete Backend Folder Structure](#section-b-complete-backend-folder-structure)
3. [Section C: Major Backend Modules](#section-c-major-backend-modules)
4. [Section D: Complete API Documentation](#section-d-complete-api-documentation)
5. [Section E: Master API Inventory Table](#section-e-master-api-inventory-table)
6. [Section F: Authentication & Authorization](#section-f-authentication--authorization)
7. [Section G: Database Architecture & Schema](#section-g-database-architecture--schema)
8. [Section H: Background Jobs, Scheduling & Concurrency](#section-h-background-jobs-scheduling--concurrency)
9. [Section I: AI, LLM, RAG & Multi-Agent Engine](#section-i-ai-llm-rag--multi-agent-engine)
10. [Section J: File & Document Processing Pipeline](#section-j-file--document-processing-pipeline)
11. [Section K: External Integrations & Model Context Protocol (MCP)](#section-k-external-integrations--model-context-protocol-mcp)
12. [Section L: Environment Variables Reference](#section-l-environment-variables-reference)
13. [Section M: Error Handling, Auditing & Observability](#section-m-error-handling-auditing--observability)
14. [Section N: Production Architecture & Deployment](#section-n-production-architecture--deployment)
15. [Section O: Troubleshooting & Operational Runbook](#section-o-troubleshooting--operational-runbook)

---

## Section A: Backend Overview

### 1. What the Backend Does
CapitalEdge AI Enterprise Backend is an enterprise-grade financial intelligence, multi-modal document research, and autonomous agent orchestration server. It bridges natural language queries with enterprise documents, live financial market data, SEC company filings, financial ratios, corporate registries, and scheduled autonomous research workflows.

### 2. Main Responsibilities
* **User & Tenant Management:** Secure authentication (Bcrypt, JWT access & refresh tokens, Google OAuth 2.0, email OTP verification), tier-based subscription management (`free`, `plus`, `pro`, `enterprise`), credit limits, and profile customizations.
* **Multi-Modal Document Processing:** End-to-end ingestion of PDFs, Office documents (Word, Excel), CSVs, text, and scanned images via a hybrid OCR pipeline (PyMuPDF native extraction + OpenCV/RapidOCR/Tesseract thread pool) with sliding-window chunking and 1536-dimensional OpenAI vector embeddings.
* **Retrieval-Augmented Generation (RAG):** Two-stage retrieval combining PostgreSQL `pgvector` HNSW cosine similarity search with an LLM listwise cross-encoder re-ranking service, dynamic query rewriting, intent classification, and automatic web search fallback.
* **Model Context Protocol (MCP) Financial Integrations:** Native connectors and a standalone FastMCP server providing real-time data from SEC EDGAR (10-K, 10-Q, 8-K facts), Roic.ai (financial ratios, valuation metrics, earnings call transcripts), UK Companies House, and Alpha Vantage (market news and sentiment).
* **Multi-Document Analysis Grids (Takeaways):** Comparative matrix generation across multiple uploaded documents with asynchronous column-by-column AI extraction and structured exports (CSV, JSON, Markdown).
* **Dynamic Skills Engine:** Creation, prompt generation, structural/AI validation, versioning, publishing, and automated injection of custom analytic skills into chat turns.
* **Autonomous Agent Workflow Engine:** Trigger-agnostic multi-step execution loop that decomposes high-level goals into sequential atomic tasks, checks and binds relevant skills, invokes specialized tools, maintains a shared memory blackboard, retries on failure, and synthesizes executive reports.
* **Background Scheduling & Worker Engine:** Lifespan-managed polling scheduler evaluating cron expressions and intervals, running automated agents, enforcing execution quotas, recovering stale jobs on startup, and emailing HTML summary reports.
* **In-Flight Generation Runtime:** Decoupled SSE token streaming architecture allowing users to stop generations midway or reconnect seamlessly without losing database state.

### 3. End-to-End Request Flow
1. **Request Intake:** The HTTP request arrives at `server/main.py`, passes through CORS headers and the global audit logging context.
2. **Routing & Authentication:** `server/api.py` dispatches the request to the matching sub-router in `server/api_v1/endpoints/`. If protected, `server/api_v1/deps.py` extracts and verifies the Bearer JWT, querying `users` to inject `current_user`.
3. **Credit Verification:** For AI operations, `CreditService` checks the user's remaining daily credit allowance and locks the user record to prevent race conditions.
4. **Execution Orchestration:**
   * **For Chat/RAG:** `ChatOrchestrationService` rewrites the query, runs intent classification in parallel via thread pools, retrieves relevant chunks from `document_chunks` using pgvector cosine distance, queries MCP tools if financial entities exist, invokes `DeepResearchOrchestrator` or `ThinkLonger` if configured, and streams SSE tokens via `generation_runtime.py`.
   * **For Workflows:** `AgentWorkflowEngine` initializes `AgentSharedState`, decomposes the prompt into steps, resolves applicable skills, executes tools with backoff retries, updates the blackboard, and records step logs in `agent_workflow_step_logs`.
   * **For Ingestion:** `DocumentProcessor` validates the file, extracts text via native PyMuPDF or OCR (RapidOCR/Tesseract), segments text into chunks with character offsets, generates OpenAI embeddings, and persists metadata and vectors to PostgreSQL.
5. **Persistence & Response:** Results are committed to PostgreSQL via the CRUD layer (`server/crud/`), credit deductions are recorded in `credit_usage_logs`, audit logs are written to `logs/audit.log`, and JSON or SSE events are returned to the client.
---

## Section B: Complete Backend Folder Structure

The backend source code is organized into a modular, layered structure:

```
agentic-ai/
├── alembic/                         # Database schema migrations
│   ├── env.py                       # Alembic runtime configuration & pgvector hook
│   ├── script.py.mako               # Migration template
│   └── versions/                    # 21 incremental migration scripts (Vv1 - Vv21)
├── logs/                            # Runtime and audit logs
│   └── audit.log                    # Structured JSON audit log stream
├── server/                          # Core application package
│   ├── api.py                       # Central router registering all /v1 endpoints
│   ├── config.py                    # Pydantic Settings & environment variables
│   ├── main.py                      # FastAPI application entrypoint & lifespan
│   ├── mcp_server.py                # Standalone FastMCP server for financial tools
│   ├── migrate_db.py                # Database migration helper script
│   ├── api_v1/                      # API routing and endpoint controllers
│   │   ├── deps.py                  # Dependency injection (get_db, get_current_user)
│   │   └── endpoints/               # Route controllers by domain
│   │       ├── agent_workflow.py    # Agent workflows CRUD, execution, scheduler APIs
│   │       ├── document_takeaways.py# Multi-document analysis grids APIs
│   │       ├── documents.py         # Document viewing, signed URLs, citations, PDF export
│   │       ├── genai.py             # Chat sessions, SSE messaging, generation stop/reconnect
│   │       ├── prompt_templates.py  # Prompt templates and default landing agents
│   │       ├── skill.py             # Skill generation, validation, CRUD, publishing
│   │       ├── user.py              # Auth, profile settings, credits, subscriptions, support
│   │       └── workspace.py         # Workspace CRUD, document uploads, document deletion
│   ├── crud/                        # Data Access Layer (SQLAlchemy CRUD operations)
│   │   ├── __init__.py              # Exported CRUD singletons
│   │   ├── base.py                  # Generic CRUDBase repository class
│   │   ├── crud_agent_workflow.py   # Agent workflows, executions, and step logs CRUD
│   │   ├── crud_chat_message.py     # Chat messages CRUD
│   │   ├── crud_chat_session.py     # Chat sessions CRUD
│   │   ├── crud_credit_usage_log.py # Credit usage logs CRUD
│   │   ├── crud_document.py         # Document metadata CRUD
│   │   ├── crud_document_chat_link.py # Document-to-chat associations CRUD
│   │   ├── crud_document_chuck.py   # Document chunks & pgvector search CRUD
│   │   ├── crud_document_takeaways.py # Takeaway grids, columns, rows, cells CRUD
│   │   ├── crud_emial_otp.py        # Email OTP CRUD
│   │   ├── crud_genai.py            # Legacy GenAI CRUD
│   │   ├── crud_help_query.py       # Support tickets CRUD
│   │   ├── crud_skill.py            # User skills CRUD
│   │   ├── crud_user.py             # User and row-locking CRUD
│   │   └── crud_workspace.py        # Workspace CRUD
│   ├── db/                          # Database connection & base classes
│   │   ├── base.py                  # Model aggregator for Alembic discovery
│   │   ├── base_class.py            # Declarative Base with table name generation
│   │   ├── init_db.py               # Table creation & vector extension initializer
│   │   └── session.py               # SQLAlchemy Engine & SessionLocal factory
│   ├── models/                      # SQLAlchemy ORM database models
│   │   ├── __init__.py              # Model exports
│   │   ├── agent_workflow.py        # AgentWorkflow, Execution, StepLog models & Enums
│   │   ├── document_takeaways.py    # TakeawayGrid, Column, Row, Cell models
│   │   ├── genai.py                 # ChatSession, ChatMessage models
│   │   ├── skill.py                 # Skill model & SkillStatus Enum
│   │   ├── user.py                  # User, EmailOtp, HelpQuery, CreditUsageLog models
│   │   └── workspace.py             # Workspace, Document, Chunk, ChatLink models
│   ├── prompt_templates/            # Built-in Landing Page Agent Prompt Templates (10 sets)
│   │   ├── brand_health_check/      # manifest.json + prompt.md
│   │   ├── deal_diligence/          # manifest.json + prompt.md
│   │   ├── launch_lab/              # manifest.json + prompt.md
│   │   ├── management_meter/        # manifest.json + prompt.md
│   │   ├── market_primer/           # manifest.json + prompt.md
│   │   ├── persona_pro/             # manifest.json + prompt.md
│   │   ├── sector_scan/             # manifest.json + prompt.md
│   │   ├── unit_economics_lab/      # manifest.json + prompt.md
│   │   ├── voice_of_customer/       # manifest.json + prompt.md
│   │   └── win_loss_review/         # manifest.json + prompt.md
│   ├── prompts/                     # LLM Prompt definitions and builders
│   │   ├── __init__.py
│   │   └── document_takeaways.py    # Prompts for matrix grid cell generation
│   ├── schemas/                     # Pydantic request, response, and DTO models
│   │   ├── __init__.py              # Schema exports
│   │   ├── agent_workflow.py        # Agent workflow schemas, plans, runs, steps DTOs
│   │   ├── credits.py               # Credit balance & usage log schemas
│   │   ├── document_takeaways.py    # Takeaway grid, column, row, cell schemas
│   │   ├── documents.py             # Document, citation detail, signed URL schemas
│   │   ├── genai.py                 # Chat session, message, and generation schemas
│   │   ├── settings.py              # User settings, profile, help query schemas
│   │   ├── skill.py                 # Skill creation, validation, suggestion schemas
│   │   ├── subscription.py          # Pricing plan tiers and upgrade schemas
│   │   ├── user.py                  # User auth, login, OTP, registration schemas
│   │   └── workspace.py             # Workspace, document ingestion, chunk schemas
│   ├── services/                    # Core business logic services
│   │   ├── agent_execution_service.py # High-level workflow execution coordinator
│   │   ├── agent_scheduler_service.py # Cron/interval scheduler & background loop
│   │   ├── agent_seed_service.py    # Canonical landing page agent template supplier
│   │   ├── agent_title_service.py   # AI chat title generator for agent sessions
│   │   ├── agent_tool_registry.py   # Tool registry (RAG, SEC, Roic, News, Web)
│   │   ├── agent_trigger_service.py # Automatic agent intent matcher & chat runner
│   │   ├── agent_validation_service.py # Agent cron, time, and parameter validator
│   │   ├── agent_workflow_engine.py # Decoupled multi-step cognitive agent engine
│   │   ├── audit_logger.py          # Centralized structured JSON audit logger
│   │   ├── chat_orchestration_service.py # Chat routing, planning & SSE pipeline
│   │   ├── chunking_service.py      # Character sliding-window text chunker
│   │   ├── citation_service.py      # Citation extraction, renumbering & formatting
│   │   ├── companies_house_mcp.py   # UK Companies House connector & caching
│   │   ├── credit_service.py        # Daily credit quota, deduction & reset service
│   │   ├── deep_research.py         # Multi-subquery recursive research orchestrator
│   │   ├── document_processor.py    # File upload ingestion orchestrator
│   │   ├── document_takeaways_service.py # Analysis matrix grid execution service
│   │   ├── embedding_service.py     # OpenAI embedding client with batching
│   │   ├── financial_news_mcp.py    # Alpha Vantage financial news connector
│   │   ├── generation_runtime.py    # In-flight generation manager & stop handler
│   │   ├── ocr_service.py           # Hybrid PyMuPDF & OpenCV/OCR text extraction
│   │   ├── pdf_export_service.py    # ReportLab styled PDF report generator
│   │   ├── pricing_service.py       # Plan tiers, limits, and pricing specifications
│   │   ├── prompt_template_registry.py # Filesystem prompt template loader & cache
│   │   ├── rag_service.py           # Two-stage RAG, query rewriting & tool routing
│   │   ├── rerank_service.py        # LLM listwise cross-encoder candidate re-ranker
│   │   ├── roic_ai_mcp.py           # Roic.ai financial ratios & transcript connector
│   │   ├── sec_edgar_mcp.py         # SEC EDGAR company facts & filings connector
│   │   ├── settings_service.py      # User settings, profile & help query service
│   │   ├── skill_generation_service.py # AI-assisted skill prompt generator
│   │   ├── skill_instruction_service.py # Skill instruction compiler & formatter
│   │   ├── skill_selection_service.py # LLM skill selector for chat and steps
│   │   ├── skill_suggestion_registry.py # Suggestion manifest loader & cache
│   │   └── skill_validation_service.py # Structural & LLM quality skill validator
│   ├── skill_suggestions/           # Pre-configured skill generation prompt chips
│   │   ├── api_webhook_integrator/  # manifest.json
│   │   ├── ascii_art_diagrams/      # manifest.json
│   │   ├── improve_skill_md/        # manifest.json
│   │   ├── pdf_report_extractor/    # manifest.json
│   │   ├── scheduled_morning_brief/ # manifest.json
│   │   └── validation_and_packaging/# manifest.json
│   └── utils/                       # Shared utility functions and helpers
│       ├── auth.py                  # Password hashing (Bcrypt) & JWT encode/decode
│       ├── chat_helpers.py          # LLM parameter helpers & routing log builders
│       ├── common.py                # String helpers & slugify
│       ├── helpers.py               # Document resolution & citation marker helpers
│       ├── llm_client.py            # OpenAI client factory & retry wrappers
│       ├── mail.py                  # FastAPI-Mail sender & HTML templates
│       ├── markdown.py              # Markdown formatting and spacing normalization
│       ├── prompt.py                # System prompt templates & agent definitions
│       ├── signed_url.py            # HMAC-SHA256 time-limited signed URL generator
│       └── source_attribution.py    # Source attribution extraction from citations
├── static/                          # Static assets and uploaded files
│   ├── avatars/                     # User avatar image uploads
│   └── workspace_docs/              # Ingested PDF, DOCX, XLSX, and TXT files
├── tests/                           # Pytest test suite
│   ├── conftest.py                  # Pytest fixtures and mock setups
│   ├── unit/                        # Unit tests for services, auth, and endpoints
│   └── ...                          # Test modules
├── Dockerfile                       # Production container build definition
├── docker-compose.yml               # Multi-container orchestration (App + Postgres + pgvector)
├── requirements.txt                 # Pinned Python package dependencies
├── serversetup.sh                   # EC2 / Linux automated server setup script
└── app.yml                          # Deployment configuration descriptor
```

---

## Section C: Major Backend Modules

### 1. Authentication, Authorization & User Management Module
* **Purpose:** Manages user identity, registration, secure login, Google OAuth 2.0 integration, email OTP verification, password resets, and role-based access control.
* **Business Explanation:** Ensures only authorized users access the platform, secures confidential research documents, and provides seamless Google single sign-on.
* **Technical Explanation:** Passwords are encrypted using Bcrypt via Passlib. Access tokens are stateless HMAC-SHA256 JWTs valid for 14 days; refresh tokens are valid for 28 days. Email verification and password reset OTPs (6-digit numeric) expire in 10 minutes with rate-limited retries (max 5 attempts).
* **Key Files:** `server/api_v1/endpoints/user.py`, `server/utils/auth.py`, `server/utils/mail.py`, `server/api_v1/deps.py`, `server/models/user.py`.
* **Key APIs:** `POST /v1/user/register`, `POST /v1/user/login`, `POST /v1/user/google/login`, `POST /v1/user/verify-otp`, `POST /v1/user/resend-otp`, `POST /v1/user/forgot-password`, `POST /v1/user/reset-password`, `POST /v1/user/refresh-token`, `GET /v1/user/me`.
* **Models:** `User`, `EmailOtp`.
* **Dependencies:** `python-jose`, `passlib[bcrypt]`, `google-auth`, `fastapi-mail`.

---

### 2. User Profile, Settings & Support Ticket Module
* **Purpose:** Allows users to configure display preferences, customize AI personality traits, set global custom instructions, toggle data retention, upload avatars, and submit support tickets.
* **Business Explanation:** Gives users full control over how AI assistants behave, what language is used, and allows direct communication with customer support.
* **Technical Explanation:** `SettingsService` serializes user preferences into PostgreSQL fields (`nickname`, `occupation`, `bio`, `ai_traits`, `custom_instructions`, `settings_json`). Support queries are stored in `help_queries` and trigger dual asynchronous email alerts via `fastapi-mail` to the support team (`hello@ojasyalabs.com`) and an acknowledgment to the user.
* **Key Files:** `server/api_v1/endpoints/user.py`, `server/services/settings_service.py`, `server/models/user.py`, `server/schemas/settings.py`.
* **Key APIs:** `GET /v1/user/settings`, `PATCH /v1/user/settings`, `POST /v1/user/avatar`, `DELETE /v1/user/avatar`, `POST /v1/user/help`, `GET /v1/user/about`.
* **Models:** `User`, `HelpQuery`.

---

### 3. Subscription Tiers & Daily Credit Quota Ledger Module
* **Purpose:** Manages subscription tier specifications (`free`, `plus`, `pro`, `enterprise`), enforces daily credit quotas, provides concurrency-safe balance deductions, records immutable transaction logs, and automatically resets credits at 00:00 UTC.
* **Business Explanation:** Monetizes platform features, prevents API abuse, and provides transparency into AI usage.
* **Technical Explanation:** `PricingService` defines canonical tier quotas (Free: 100, Plus: 1,000, Pro: 5,000, Enterprise: 25,000 credits/day). `CreditService` performs atomic balance deductions using PostgreSQL row-level locks (`with_for_update`) to prevent race conditions during parallel requests, logging every transaction to `credit_usage_logs`.
* **Key Files:** `server/services/pricing_service.py`, `server/services/credit_service.py`, `server/models/user.py`, `server/schemas/credits.py`, `server/schemas/subscription.py`.
* **Key APIs:** `GET /v1/user/plans`, `GET /v1/user/subscription`, `POST /v1/user/upgrade`, `GET /v1/user/credits`, `POST /v1/user/credits/consume`, `POST /v1/user/credits/reset`, `GET /v1/user/credits/history`.
* **Models:** `User`, `CreditUsageLog`.

---

### 4. Workspaces & Document Ingestion Pipeline Module
* **Purpose:** Manages multi-tenant workspaces, file uploads (PDF, DOCX, XLSX, CSV, images), text extraction, semantic chunking, and 1536-dimensional vector embedding persistence.
* **Business Explanation:** Allows analysts to organize documents by project or company and search across their contents instantly.
* **Technical Explanation:** Files are saved to `static/workspace_docs/`. `OCRService` uses a hybrid pipeline: PyMuPDF extracts native digital text; embedded or scanned images are preprocessed with OpenCV and OCR'd via RapidOCR/Tesseract. `ChunkingService` applies a character-based sliding window (default 1000 chars, 200 overlap), recording character offsets and page numbers. `EmbeddingService` batches chunk text to OpenAI `text-embedding-3-small` (1536 dimensions) and saves to `document_chunks` with an HNSW cosine index (`ix_document_chunks_embedding`).
* **Key Files:** `server/api_v1/endpoints/workspace.py`, `server/services/document_processor.py`, `server/services/ocr_service.py`, `server/services/chunking_service.py`, `server/services/embedding_service.py`, `server/models/workspace.py`.
* **Key APIs:** `POST /v1/workspaces`, `GET /v1/workspaces`, `GET /v1/workspaces/{id}`, `PATCH /v1/workspaces/{id}`, `DELETE /v1/workspaces/{id}`, `GET /v1/workspaces/public/default`, `POST /v1/workspaces/{id}/documents`, `GET /v1/workspaces/{id}/documents`, `DELETE /v1/workspaces/{id}/documents/{doc_id}`.
* **Models:** `Workspace`, `Document`, `DocumentChunk`.

---

### 5. Document Viewer, Citations & PDF Report Export Module
* **Purpose:** Provides secure document streaming, visual highlight rectangle overlays for PDF pages, citation verification, and downloadable styled PDF reports.
* **Business Explanation:** Allows analysts to verify AI assertions against original document text and download audit-ready executive reports.
* **Technical Explanation:** `signed_url.py` generates HMAC-SHA256 time-limited tokens for direct file streaming. `CitationService` extracts citations from LLM responses, maps them to `DocumentChunk` records or web URLs, and normalizes them into `[1..k]` sequences. The highlights endpoint searches PDF pages using PyMuPDF (`fitz`) and returns normalized bounding box coordinates `(x0, y0, x1, y1)` for frontend PDF rendering. `PDFExportService` compiles reports with ReportLab, embedding tables, citations, and CapitalEdge branding.
* **Key Files:** `server/api_v1/endpoints/documents.py`, `server/services/citation_service.py`, `server/services/pdf_export_service.py`, `server/utils/signed_url.py`.
* **Key APIs:** `GET /v1/citations/{citation_id}`, `GET /v1/documents/{document_id}/viewer`, `GET /v1/documents/{document_id}/highlights`, `GET /v1/documents/files/{file_path:path}`, `POST /v1/documents/export/pdf`, `GET /v1/documents/roic/transcript`.
* **Dependencies:** `reportlab`, `pymupdf` (`fitz`).

---

### 6. AI Chat & Intelligent RAG Orchestration Module
* **Purpose:** Handles conversational multi-turn AI interactions, query rewriting, multi-source retrieval (documents, web, SEC, Roic, news), Deep Research, Think Longer reasoning, and token streaming via SSE.
* **Business Explanation:** Provides an interactive financial AI research assistant capable of answering complex analytical questions with cited sources.
* **Technical Explanation:** `ChatOrchestrationService` coordinates the pipeline:
  1. Rewrites queries based on chat history.
  2. Runs parallel intent classification in `ThreadPoolExecutor`.
  3. Evaluates tool routing (RAG vs SEC vs Roic vs News vs Web Search).
  4. Executes two-stage RAG: pgvector cosine search + LLM listwise re-ranking (`RerankService`).
  5. Performs sufficiency checks (DEF-011, DEF-012) with automatic web search fallback.
  6. Supports Deep Research (sub-query generation) and Think Longer (reasoning effort).
  7. Registers the in-flight generation in `generation_runtime.py` and streams tokens as SSE frames (`data: {"type": "token", "token": "..."}`).
* **Key Files:** `server/api_v1/endpoints/genai.py`, `server/services/chat_orchestration_service.py`, `server/services/rag_service.py`, `server/services/rerank_service.py`, `server/services/deep_research.py`, `server/services/generation_runtime.py`, `server/models/genai.py`.
* **Key APIs:** `POST /v1/ai/chats`, `GET /v1/ai/chats`, `GET /v1/ai/chats/{chat_id}`, `DELETE /v1/ai/chats/{chat_id}`, `GET /v1/ai/chats/public/default`, `POST /v1/ai/chats/{chat_id}/messages`, `GET /v1/ai/chats/{chat_id}/messages`, `GET /v1/ai/generations/active`, `GET /v1/ai/generations/{generation_id}/stream`, `POST /v1/ai/chats/{chat_id}/stop`, `POST /v1/ai/generations/{generation_id}/stop`.
* **Models:** `ChatSession`, `ChatMessage`, `DocumentChatLink`.

---

### 7. Document Takeaways Matrix Grid Module
* **Purpose:** Generates structured comparative analysis grids across multiple documents, extracting executive summaries, key takeaways, and custom user-defined criteria.
* **Business Explanation:** Enables instant cross-document comparison (e.g., comparing Q3 earnings across three competing companies).
* **Technical Explanation:** A grid creates a row per document and columns (`Document Name`, `Expert Description`, `Key Takeaways`, or custom columns). Cell tasks are processed in the background (`process_grid_cell_task`), retrieving document chunks via RAG and querying the LLM for structured JSON answers. Supports cell retries, grid duplication, renaming, and export to CSV, JSON, and Markdown.
* **Key Files:** `server/api_v1/endpoints/document_takeaways.py`, `server/services/document_takeaways_service.py`, `server/models/document_takeaways.py`, `server/prompts/document_takeaways.py`.
* **Key APIs:** `POST /v1/workspaces/{workspace_id}/takeaways`, `GET /v1/workspaces/{workspace_id}/takeaways`, `GET /v1/workspaces/{workspace_id}/takeaways/{grid_id}`, `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/columns`, `DELETE /v1/workspaces/{workspace_id}/takeaways/{grid_id}/columns/{column_id}`, `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/documents`, `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/cells/{cell_id}/retry`, `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/duplicate`, `PATCH /v1/workspaces/{workspace_id}/takeaways/{grid_id}/rename`, `DELETE /v1/workspaces/{workspace_id}/takeaways/{grid_id}`, `GET /v1/workspaces/{workspace_id}/takeaways/{grid_id}/export`, `GET /v1/document-takeaways`.
* **Models:** `DocumentTakeawayGrid`, `DocumentTakeawayGridColumn`, `DocumentTakeawayGridRow`, `DocumentTakeawayGridCell`.

---

### 8. Dynamic Skills Management Module
* **Purpose:** Allows users to create, generate via AI, validate, test, publish, and inject specialized domain instructions into AI workflows.
* **Business Explanation:** Enables organizations to codify internal methodologies (e.g., specific valuation models, audit checklists) into reusable AI capabilities.
* **Technical Explanation:** Skills are stored with draft or published status (`SkillStatus`). `SkillGenerationService` drafts complete skill instructions based on brief user prompts. `SkillValidationService` executes structural checks and LLM quality reviews (score 0-100). During chat, `SkillSelectionService` uses vector matching and LLM intent matching to select up to 5 published skills and injects their formatted instructions into the prompt context.
* **Key Files:** `server/api_v1/endpoints/skill.py`, `server/services/skill_generation_service.py`, `server/services/skill_validation_service.py`, `server/services/skill_selection_service.py`, `server/services/skill_instruction_service.py`, `server/services/skill_suggestion_registry.py`, `server/models/skill.py`.
* **Key APIs:** `GET /v1/skills/suggestions`, `POST /v1/skills/generate-draft`, `POST /v1/skills/validate-draft`, `POST /v1/skills`, `GET /v1/skills`, `GET /v1/skills/{skill_id}`, `PATCH /v1/skills/{skill_id}`, `POST /v1/skills/{skill_id}/publish`, `DELETE /v1/skills/{skill_id}`.
* **Models:** `Skill`.

---

### 9. Autonomous Agent Workflow Engine & Scheduler Module
* **Purpose:** Orchestrates multi-step autonomous AI workflows triggered on-demand or on a recurring schedule (daily, weekly, monthly, cron), executing plans, logging step-by-step traces, and dispatching summary emails.
* **Business Explanation:** Automates recurring financial research (e.g., "Every Monday at 8 AM, analyze competitor SEC filings and email me a brief").
* **Technical Explanation:**
  * **Engine (`AgentWorkflowEngine`):** Decomposes tasks into 1-4 atomic steps using LLM planning, matches skills, runs tools via `AgentToolRegistry`, stores intermediate variables in `AgentSharedState`, applies exponential backoff retries (up to 2 retries), logs each step to `agent_workflow_step_logs`, and synthesizes final reports with citations.
  * **Scheduler (`AgentSchedulerService`):** Starts on FastAPI lifespan, polling every 30 seconds. Evaluates cron and time fields against local timezones, enforces concurrency locks and execution quotas (`scheduled_count_limit`), updates `next_run_at`, creates daily chat sessions, and sends styled HTML emails via `fastapi-mail`.
* **Key Files:** `server/api_v1/endpoints/agent_workflow.py`, `server/services/agent_workflow_engine.py`, `server/services/agent_scheduler_service.py`, `server/services/agent_execution_service.py`, `server/services/agent_trigger_service.py`, `server/services/agent_tool_registry.py`, `server/services/agent_validation_service.py`, `server/services/agent_seed_service.py`, `server/models/agent_workflow.py`.
* **Key APIs:** `POST /v1/agents/validate`, `POST /v1/agents`, `GET /v1/agents`, `GET /v1/agents/scheduler/status`, `POST /v1/agents/scheduler/tick`, `GET /v1/agents/tools`, `GET /v1/agents/templates`, `POST /v1/agents/run`, `GET /v1/agents/runs/{run_id}`, `GET /v1/agents/runs/{run_id}/steps`, `GET /v1/agents/{agent_id}`, `PATCH /v1/agents/{agent_id}`, `POST /v1/agents/{agent_id}/activate`, `POST /v1/agents/{agent_id}/pause`, `POST /v1/agents/{agent_id}/deactivate`, `DELETE /v1/agents/{agent_id}`, `POST /v1/agents/{agent_id}/run`, `POST /v1/agents/{agent_id}/trigger-schedule`, `GET /v1/agents/{agent_id}/runs`.
* **Models:** `AgentWorkflow`, `AgentWorkflowExecution`, `AgentWorkflowStepLog`.

---

### 10. Prompt Template Registry & Landing Agent Seeds Module
* **Purpose:** Supplies 10 pre-configured canonical Landing Page Agent templates representing core financial workflows.
* **Business Explanation:** Provides instant value to users with one-click templates for market research, governance, unit economics, and M&A diligence.
* **Technical Explanation:** `PromptTemplateRegistry` discovers filesystem templates in `server/prompt_templates/` containing `manifest.json` and `prompt.md`. Synchronized with `AgentSeedService` to supply the 10 canonical templates:
  1. Market Primer (`market_primer`)
  2. Management Meter (`management_meter`)
  3. Deal Diligence (`deal_diligence`)
  4. Unit Economics Lab (`unit_economics_lab`)
  5. Sector Scan (`sector_scan`)
  6. Brand Health Check (`brand_health_check`)
  7. Win Loss Review (`win_loss_review`)
  8. Voice of Customer (`voice_of_customer`)
  9. Persona Pro (`persona_pro`)
  10. Launch Lab (`launch_lab`)
* **Key Files:** `server/api_v1/endpoints/prompt_templates.py`, `server/services/prompt_template_registry.py`, `server/services/agent_seed_service.py`.
* **Key APIs:** `GET /v1/cards/prompt-templates`.

---

### 11. Model Context Protocol (MCP) & Financial Tools Module
* **Purpose:** Exposes external financial data connectors directly to RAG and Agent workflows, as well as over standard MCP transports (stdio / SSE) via FastMCP.
* **Business Explanation:** Grounds AI answers in audited company filings, consensus financial metrics, and official registry filings.
* **Technical Explanation:**
  * `sec_edgar_mcp.py`: Resolves company tickers to CIKs, queries SEC company facts API, and formats 10-K/10-Q metrics.
  * `roic_ai_mcp.py`: Fetches valuation multiples, financial statements, and transcript text.
  * `companies_house_mcp.py`: Queries UK government registry for company profile and officer records.
  * `financial_news_mcp.py`: Queries Alpha Vantage News & Sentiment API for market headlines.
  * `mcp_server.py`: FastMCP server (`capitaledge-financial-data`) exposing tools to external MCP agents.
* **Key Files:** `server/mcp_server.py`, `server/services/sec_edgar_mcp.py`, `server/services/roic_ai_mcp.py`, `server/services/companies_house_mcp.py`, `server/services/financial_news_mcp.py`.
* **Dependencies:** `fastmcp`, `httpx`, `serpapi`.

---

### 12. Audit Logging, Diagnostics & Observability Module
* **Purpose:** Emits structured JSON events for all platform operations, API calls, LLM latencies, and OCR processing.
* **Business Explanation:** Provides complete auditability for compliance, debugging, and system performance monitoring.
* **Technical Explanation:** `AuditLogger` writes structured JSON events to `logs/audit.log` and stdout. Captures timestamps, user ID, workspace ID, chat ID, action names (`CHAT_ORCHESTRATION`, `DOCUMENT_OCR`, `CANDIDATE_RERANKING`), processing times, and custom metadata. Global exception handlers catch unhandled errors and format standardized JSON error responses.
* **Key Files:** `server/services/audit_logger.py`, `server/main.py`.

---

## Section D: Complete API Documentation

Every endpoint in CapitalEdge AI Enterprise Backend follows standard REST conventions and returns a consistent JSON envelope (or SSE stream):

```json
{
  "success": true,
  "error": null,
  "data": { ... },
  "message": "Human readable description."
}
```

---

### 1. Authentication & User APIs (`server/api_v1/endpoints/user.py`)

#### `POST /v1/user/register`
* **Purpose:** Register a new user account with email and password. Generates a 6-digit OTP and sends a verification email.
* **Auth:** Public
* **Body:**
  ```json
  {
    "email": "analyst@example.com",
    "password": "SecurePassword123!",
    "full_name": "Jane Doe"
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "success": true,
    "error": null,
    "data": {
      "id": "u-12345",
      "email": "analyst@example.com",
      "full_name": "Jane Doe",
      "is_active": false
    },
    "message": "Registration successful. Please verify your email with the OTP sent."
  }
  ```
* **Status Codes:** `200` (Success), `400` (Email already registered or invalid), `422` (Validation error).

#### `POST /v1/user/login`
* **Purpose:** Authenticate an existing verified user with email and password.
* **Auth:** Public
* **Body:**
  ```json
  {
    "email": "analyst@example.com",
    "password": "SecurePassword123!"
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "success": true,
    "error": null,
    "data": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5c...",
      "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5c...",
      "token_type": "bearer",
      "user": {
        "id": "u-12345",
        "email": "analyst@example.com",
        "full_name": "Jane Doe",
        "subscription_tier": "free",
        "credits_remaining": 100
      }
    },
    "message": "Login successful."
  }
  ```
* **Status Codes:** `200` (Success), `400` (Invalid credentials or unverified email).

#### `POST /v1/user/google/login`
* **Purpose:** Authenticate or register a user using a Google OAuth 2.0 ID token.
* **Auth:** Public
* **Body:**
  ```json
  {
    "credential": "eyJhbGciOiJSUzI1NiIsImtpZCI6..."
  }
  ```
* **Response (200 OK):** Same token payload as `/login`.

#### `POST /v1/user/verify-otp`
* **Purpose:** Verify the 6-digit email OTP for account activation or password resets.
* **Auth:** Public
* **Body:**
  ```json
  {
    "email": "analyst@example.com",
    "otp": "482910"
  }
  ```
* **Response (200 OK):** `{"success": true, "message": "Email verified successfully."}`

#### `POST /v1/user/resend-otp`
* **Purpose:** Generate and resend a fresh 6-digit OTP to the user's email.
* **Auth:** Public
* **Body:** `{"email": "analyst@example.com"}`

#### `POST /v1/user/forgot-password`
* **Purpose:** Initiate password recovery by sending a reset OTP.
* **Auth:** Public
* **Body:** `{"email": "analyst@example.com"}`

#### `POST /v1/user/reset-password`
* **Purpose:** Reset password using the verified OTP code.
* **Auth:** Public
* **Body:**
  ```json
  {
    "email": "analyst@example.com",
    "otp": "482910",
    "new_password": "NewSecurePassword456!"
  }
  ```

#### `POST /v1/user/refresh-token`
* **Purpose:** Exchange a valid refresh token for a new access token.
* **Auth:** Public (Refresh Token in body)
* **Body:** `{"refresh_token": "eyJhbGciOiJIUzI1NiIsInR5c..."}`

#### `GET /v1/user/me`
* **Purpose:** Retrieve the profile and credit status of the authenticated user.
* **Auth:** Bearer JWT
* **Response (200 OK):** Returns full `UserResponse` object with subscription tier and daily credits.

#### `GET /v1/user/settings`
* **Purpose:** Retrieve complete user profile preferences, AI traits, and connector states.
* **Auth:** Bearer JWT
* **Response (200 OK):** Returns `UserSettings` DTO.

#### `PATCH /v1/user/settings`
* **Purpose:** Update profile preferences (nickname, bio, occupation, language, custom instructions, AI traits).
* **Auth:** Bearer JWT
* **Body:** Partial `UserSettingsUpdate` object.

#### `POST /v1/user/avatar`
* **Purpose:** Upload a user profile picture (PNG, JPG, WEBP; max 5MB).
* **Auth:** Bearer JWT
* **Body:** `multipart/form-data` with `file`.

#### `DELETE /v1/user/avatar`
* **Purpose:** Remove custom avatar and revert to default.
* **Auth:** Bearer JWT

#### `POST /v1/user/help`
* **Purpose:** Submit a support query ticket. Dispatches email to support and user.
* **Auth:** Bearer JWT
* **Body:**
  ```json
  {
    "subject": "Question about SEC Edgar 10-K parsing",
    "message": "How do I extract balance sheet metrics for FY2025?"
  }
  ```

#### `GET /v1/user/plans`
* **Purpose:** Get all available subscription tiers, feature comparison matrices, and limits.
* **Auth:** Public

#### `GET /v1/user/subscription`
* **Purpose:** Get the current user's active subscription tier, limits, and expiration date.
* **Auth:** Bearer JWT

#### `POST /v1/user/upgrade`
* **Purpose:** Upgrade or change the user's subscription tier (`free`, `plus`, `pro`, `enterprise`).
* **Auth:** Bearer JWT
* **Body:** `{"tier": "plus", "billing_cycle": "monthly"}`

#### `GET /v1/user/credits`
* **Purpose:** Get current credit balance, usage today, and countdown to 00:00 UTC reset.
* **Auth:** Bearer JWT

#### `POST /v1/user/credits/consume`
* **Purpose:** Atomically deduct credits for an operation with an audit reason.
* **Auth:** Bearer JWT
* **Body:** `{"amount": 5, "action_type": "deep_research", "description": "Deep research query"}`

#### `GET /v1/user/credits/history`
* **Purpose:** Retrieve paginated transaction ledger entries from `credit_usage_logs`.
* **Auth:** Bearer JWT
* **Query Params:** `skip` (int, default 0), `limit` (int, default 50).

---

### 2. Workspace & Ingestion APIs (`server/api_v1/endpoints/workspace.py`)

#### `POST /v1/workspaces`
* **Purpose:** Create a new workspace folder for grouping research documents.
* **Auth:** Bearer JWT
* **Body:** `{"name": "Q1 2026 Earnings Diligence", "description": "Tech sector analysis"}`

#### `GET /v1/workspaces`
* **Purpose:** List all workspaces owned by the authenticated user with document counts.
* **Auth:** Bearer JWT

#### `GET /v1/workspaces/{workspace_id}`
* **Purpose:** Fetch a single workspace by ID.
* **Auth:** Bearer JWT

#### `PATCH /v1/workspaces/{workspace_id}`
* **Purpose:** Update workspace name or description.
* **Auth:** Bearer JWT

#### `DELETE /v1/workspaces/{workspace_id}`
* **Purpose:** Delete a workspace and cascade-delete all its documents, chunks, and analysis grids.
* **Auth:** Bearer JWT

#### `GET /v1/workspaces/public/default`
* **Purpose:** Retrieve or create the default global public sandbox workspace.
* **Auth:** Public / Optional JWT

#### `POST /v1/workspaces/{workspace_id}/documents`
* **Purpose:** Upload and ingest a document into the workspace (triggers OCR, chunking, embedding).
* **Auth:** Bearer JWT
* **Body:** `multipart/form-data` with `file` (PDF, DOCX, XLSX, CSV, TXT, MD, PNG, JPG; max 10MB).
* **Response (200 OK):** Returns created `DocumentResponse` with `storage_path` and ID.

#### `GET /v1/workspaces/{workspace_id}/documents`
* **Purpose:** List all ingested documents in a workspace.
* **Auth:** Bearer JWT

#### `DELETE /v1/workspaces/{workspace_id}/documents/{document_id}`
* **Purpose:** Delete an ingested document and remove all its vector chunks from `document_chunks`.
* **Auth:** Bearer JWT

---

### 3. Document Viewing & Citation APIs (`server/api_v1/endpoints/documents.py`)

#### `GET /v1/citations/{citation_id}`
* **Purpose:** Retrieve citation details, excerpt text, source document metadata, or external URLs.
* **Auth:** Bearer JWT

#### `GET /v1/documents/{document_id}/viewer`
* **Purpose:** Get a signed, time-limited URL for embedded PDF/document browser viewing.
* **Auth:** Bearer JWT

#### `GET /v1/documents/{document_id}/highlights`
* **Purpose:** Return bounding box rectangle coordinates on specific PDF pages matching citation text.
* **Auth:** Bearer JWT
* **Query Params:** `query` (str), `page` (int).

#### `GET /v1/documents/files/{file_path:path}`
* **Purpose:** Stream the actual file bytes using a valid signed token (`?token=...`).
* **Auth:** Signed URL Token verification

#### `POST /v1/documents/export/pdf`
* **Purpose:** Generate and download a styled PDF executive report from markdown content.
* **Auth:** Bearer JWT
* **Body:**
  ```json
  {
    "title": "Microsoft Acquisition Diligence",
    "content": "Executive report content in markdown...",
    "source_documents": ["MSFT_10K.pdf", "ATVI_Filing.pdf"]
  }
  ```

#### `GET /v1/documents/roic/transcript`
* **Purpose:** Direct endpoint to search and fetch public company earnings call transcripts from Roic.ai.
* **Auth:** Bearer JWT
* **Query Params:** `query` (str, ticker or company name).

---

### 4. AI Chat & RAG Messaging APIs (`server/api_v1/endpoints/genai.py`)

#### `POST /v1/ai/chats`
* **Purpose:** Create a new conversational chat session, optionally bound to a workspace.
* **Auth:** Bearer JWT
* **Body:** `{"title": "Meta Capex Review", "workspace_id": "ws-123", "document_ids": ["doc-1"]}`

#### `GET /v1/ai/chats`
* **Purpose:** List user's chat sessions, sorted by last updated timestamp.
* **Auth:** Bearer JWT

#### `GET /v1/ai/chats/{chat_id}`
* **Purpose:** Fetch a single chat session metadata.
* **Auth:** Bearer JWT

#### `DELETE /v1/ai/chats/{chat_id}`
* **Purpose:** Delete a chat session and all its messages.
* **Auth:** Bearer JWT

#### `GET /v1/ai/chats/public/default`
* **Purpose:** Retrieve the default public sandbox chat session.
* **Auth:** Public / Optional JWT

#### `GET /v1/ai/chats/{chat_id}/messages`
* **Purpose:** Retrieve all historical message turns for a chat session.
* **Auth:** Bearer JWT

#### `POST /v1/ai/chats/{chat_id}/messages`
* **Purpose:** Post a user query and stream the AI response via Server-Sent Events (SSE).
* **Auth:** Bearer JWT
* **Headers:** `Accept: text/event-stream`
* **Body:**
  ```json
  {
    "content": "What was Microsoft's cloud revenue growth in FY2025?",
    "mode": "auto",
    "skill_ids": ["skill-1"],
    "document_ids": ["doc-123"]
  }
  ```
* **SSE Stream Events:**
  * `data: {"type": "generation_start", "generation_id": "gen-abc"}`
  * `data: {"type": "token", "token": "Microsoft's "}`
  * `data: {"type": "token", "token": "cloud revenue..."}`
  * `data: {"type": "complete", "data": { ... full message object with citations ... }}`

#### `GET /v1/ai/generations/active`
* **Purpose:** Inspect any currently running AI generation for the authenticated user.
* **Auth:** Bearer JWT

#### `GET /v1/ai/generations/{generation_id}/stream`
* **Purpose:** Re-subscribe to an ongoing SSE generation stream after a browser refresh.
* **Auth:** Bearer JWT

#### `POST /v1/ai/chats/{chat_id}/stop` & `POST /v1/ai/generations/{generation_id}/stop`
* **Purpose:** Gracefully cancel an in-flight generation. Persists partially generated tokens and halts model calls immediately.
* **Auth:** Bearer JWT

---

### 5. Document Takeaways Matrix Grid APIs (`server/api_v1/endpoints/document_takeaways.py`)

#### `POST /v1/workspaces/{workspace_id}/takeaways`
* **Purpose:** Create an analysis grid for selected documents and kick off async cell generation.
* **Auth:** Bearer JWT
* **Body:** `{"name": "Competitor Analysis", "document_ids": ["doc-1", "doc-2"]}`

#### `GET /v1/workspaces/{workspace_id}/takeaways`
* **Purpose:** List all takeaway grids in a workspace.
* **Auth:** Bearer JWT

#### `GET /v1/workspaces/{workspace_id}/takeaways/{grid_id}`
* **Purpose:** Fetch complete grid structure with columns, rows, cell values, and status.
* **Auth:** Bearer JWT

#### `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/columns`
* **Purpose:** Add a custom AI extraction column with user-defined prompt instructions.
* **Auth:** Bearer JWT
* **Body:** `{"name": "ESG Risk Summary", "instruction": "Extract all ESG and environmental risk factors."}`

#### `DELETE /v1/workspaces/{workspace_id}/takeaways/{grid_id}/columns/{column_id}`
* **Purpose:** Delete a column and its associated cell data.
* **Auth:** Bearer JWT

#### `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/documents`
* **Purpose:** Add additional documents as new rows to an existing grid.
* **Auth:** Bearer JWT
* **Body:** `{"document_ids": ["doc-3"]}`

#### `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/cells/{cell_id}/retry`
* **Purpose:** Retry AI generation for a single failed or pending cell.
* **Auth:** Bearer JWT

#### `POST /v1/workspaces/{workspace_id}/takeaways/{grid_id}/duplicate`
* **Purpose:** Clone a grid with all its column definitions.
* **Auth:** Bearer JWT

#### `PATCH /v1/workspaces/{workspace_id}/takeaways/{grid_id}/rename`
* **Purpose:** Rename an existing takeaway grid.
* **Auth:** Bearer JWT
* **Body:** `{"name": "Q1 Strategic Review"}`

#### `DELETE /v1/workspaces/{workspace_id}/takeaways/{grid_id}`
* **Purpose:** Delete a takeaway grid.
* **Auth:** Bearer JWT

#### `GET /v1/workspaces/{workspace_id}/takeaways/{grid_id}/export`
* **Purpose:** Export the grid table to `csv`, `json`, or `markdown`.
* **Auth:** Bearer JWT
* **Query Params:** `format` (`csv` | `json` | `markdown`).

#### `GET /v1/document-takeaways`
* **Purpose:** List all takeaway grids across all workspaces owned by the user.
* **Auth:** Bearer JWT

---

### 6. Dynamic Skills APIs (`server/api_v1/endpoints/skill.py`)

#### `GET /v1/skills/suggestions`
* **Purpose:** Retrieve pre-configured skill prompt suggestion chips.
* **Auth:** Bearer JWT

#### `POST /v1/skills/generate-draft`
* **Purpose:** Use AI to generate a complete skill name, description, trigger description, and instructions from a brief concept.
* **Auth:** Bearer JWT
* **Body:** `{"prompt": "A skill that performs discounted cash flow (DCF) checks on financial statements."}`

#### `POST /v1/skills/validate-draft`
* **Purpose:** Perform structural and AI quality checks on draft skill instructions.
* **Auth:** Bearer JWT
* **Body:** `{"name": "DCF Analyzer", "instructions": "Step 1: Extract WACC..."}`

#### `POST /v1/skills`
* **Purpose:** Save a new custom skill (in draft or published status).
* **Auth:** Bearer JWT
* **Body:** `{"name": "DCF Analyzer", "description": "Analyzes cash flow", "trigger_description": "When user asks for DCF", "instructions": "...", "status": "draft"}`

#### `GET /v1/skills`
* **Purpose:** List all skills created by the authenticated user.
* **Auth:** Bearer JWT
* **Query Params:** `status` (`draft` | `published`).

#### `GET /v1/skills/{skill_id}`
* **Purpose:** Fetch a single skill by ID.
* **Auth:** Bearer JWT

#### `PATCH /v1/skills/{skill_id}`
* **Purpose:** Update skill instructions, trigger description, or metadata.
* **Auth:** Bearer JWT

#### `POST /v1/skills/{skill_id}/publish`
* **Purpose:** Validate and publish a skill, making it available for active use in chats.
* **Auth:** Bearer JWT

#### `DELETE /v1/skills/{skill_id}`
* **Purpose:** Delete a custom skill.
* **Auth:** Bearer JWT

---

### 7. Autonomous Agent Workflow APIs (`server/api_v1/endpoints/agent_workflow.py`)

#### `POST /v1/agents/validate`
* **Purpose:** Validate agent schedule configuration, cron expressions, time windows, and tool choices before saving.
* **Auth:** Bearer JWT
* **Body:**
  ```json
  {
    "name": "Weekly Competitor Tracker",
    "trigger_type": "scheduled",
    "schedule_frequency": "weekly",
    "schedule_day": "monday",
    "schedule_time": "09:00",
    "schedule_timezone": "America/New_York",
    "selected_sources": ["sec_edgar", "financial_news"]
  }
  ```

#### `POST /v1/agents`
* **Purpose:** Create and persist a new Agent Workflow.
* **Auth:** Bearer JWT
* **Body:** `AgentWorkflowCreate` schema.

#### `GET /v1/agents`
* **Purpose:** List all Agent Workflows owned by the user.
* **Auth:** Bearer JWT
* **Query Params:** `workspace_id`, `trigger_type`, `status`.

#### `GET /v1/agents/scheduler/status`
* **Purpose:** Inspect background scheduler heartbeat, active running tasks, and registered schedules.
* **Auth:** Bearer JWT

#### `POST /v1/agents/scheduler/tick`
* **Purpose:** Manually trigger a scheduler evaluation cycle (evaluates and fires due jobs immediately).
* **Auth:** Bearer JWT

#### `GET /v1/agents/tools`
* **Purpose:** List all available tools that can be bound to agents (`rag`, `sec_edgar`, `roic_ai`, `financial_news`, `companies_house`, `web_search`).
* **Auth:** Bearer JWT

#### `GET /v1/agents/templates`
* **Purpose:** Retrieve the 10 landing page agent template definitions.
* **Auth:** Bearer JWT

#### `POST /v1/agents/run`
* **Purpose:** Execute an ad-hoc, unpersisted agent workflow on demand.
* **Auth:** Bearer JWT
* **Body:** Ad-hoc execution parameters.

#### `GET /v1/agents/runs/{run_id}`
* **Purpose:** Retrieve execution details, final synthesized response, citations, and execution time for a specific run.
* **Auth:** Bearer JWT

#### `GET /v1/agents/runs/{run_id}/steps`
* **Purpose:** Fetch step-by-step cognitive execution logs and tool outputs for a specific run.
* **Auth:** Bearer JWT

#### `GET /v1/agents/{agent_id}`
* **Purpose:** Fetch an Agent Workflow configuration by ID.
* **Auth:** Bearer JWT

#### `PATCH /v1/agents/{agent_id}`
* **Purpose:** Update agent instructions, schedule, tools, or documents.
* **Auth:** Bearer JWT

#### `POST /v1/agents/{agent_id}/activate`
* **Purpose:** Activate an agent workflow and calculate its next scheduled run time.
* **Auth:** Bearer JWT

#### `POST /v1/agents/{agent_id}/pause`
* **Purpose:** Pause an active agent workflow.
* **Auth:** Bearer JWT

#### `POST /v1/agents/{agent_id}/deactivate`
* **Purpose:** Deactivate an agent workflow.
* **Auth:** Bearer JWT

#### `DELETE /v1/agents/{agent_id}`
* **Purpose:** Delete an agent workflow and its execution history.
* **Auth:** Bearer JWT

#### `POST /v1/agents/{agent_id}/run`
* **Purpose:** Manually trigger an immediate execution of an existing agent workflow.
* **Auth:** Bearer JWT

#### `POST /v1/agents/{agent_id}/trigger-schedule`
* **Purpose:** Force execution of a scheduled agent workflow using its scheduled trigger flow.
* **Auth:** Bearer JWT

#### `GET /v1/agents/{agent_id}/runs`
* **Purpose:** List past execution history and runs for a specific agent workflow.
* **Auth:** Bearer JWT

---

### 8. Prompt Template APIs (`server/api_v1/endpoints/prompt_templates.py`)

#### `GET /v1/cards/prompt-templates`
* **Purpose:** Retrieve active landing page prompt templates discovered from the filesystem (`server/prompt_templates/`).
* **Auth:** Public

---

## Section E: Master API Inventory Table

| Module | HTTP Method | Path | Auth Required | Description |
|---|---|---|---|---|
| **User & Auth** | POST | `/v1/user/register` | No | Register new account & send OTP |
| **User & Auth** | POST | `/v1/user/login` | No | Authenticate with email/password |
| **User & Auth** | POST | `/v1/user/google/login` | No | Authenticate with Google ID token |
| **User & Auth** | POST | `/v1/user/verify-otp` | No | Verify 6-digit email OTP |
| **User & Auth** | POST | `/v1/user/resend-otp` | No | Resend 6-digit email OTP |
| **User & Auth** | POST | `/v1/user/forgot-password` | No | Request password reset OTP |
| **User & Auth** | POST | `/v1/user/reset-password` | No | Reset password with OTP |
| **User & Auth** | POST | `/v1/user/refresh-token` | No | Exchange refresh token for access token |
| **User & Auth** | GET | `/v1/user/me` | Bearer JWT | Get current user profile & credit balance |
| **User & Settings** | GET | `/v1/user/settings` | Bearer JWT | Get user profile & AI customization settings |
| **User & Settings** | PATCH | `/v1/user/settings` | Bearer JWT | Update user profile & customization settings |
| **User & Settings** | POST | `/v1/user/avatar` | Bearer JWT | Upload custom profile avatar |
| **User & Settings** | DELETE | `/v1/user/avatar` | Bearer JWT | Remove custom profile avatar |
| **User & Support** | POST | `/v1/user/help` | Bearer JWT | Submit customer support ticket |
| **User & Support** | GET | `/v1/user/about` | No | Get platform & company details |
| **Subscription** | GET | `/v1/user/plans` | No | List all pricing plans and feature limits |
| **Subscription** | GET | `/v1/user/subscription` | Bearer JWT | Get user's active subscription tier & quota |
| **Subscription** | POST | `/v1/user/upgrade` | Bearer JWT | Upgrade or change subscription plan tier |
| **Credits** | GET | `/v1/user/credits` | Bearer JWT | Get credit balance and reset countdown |
| **Credits** | POST | `/v1/user/credits/consume` | Bearer JWT | Atomically deduct credits with audit reason |
| **Credits** | POST | `/v1/user/credits/reset` | Bearer JWT | Force credit reset (admin / testing) |
| **Credits** | GET | `/v1/user/credits/history` | Bearer JWT | Get credit transaction ledger history |
| **Workspaces** | POST | `/v1/workspaces` | Bearer JWT | Create a new workspace |
| **Workspaces** | GET | `/v1/workspaces` | Bearer JWT | List all user workspaces |
| **Workspaces** | GET | `/v1/workspaces/{workspace_id}` | Bearer JWT | Get a single workspace by ID |
| **Workspaces** | PATCH | `/v1/workspaces/{workspace_id}` | Bearer JWT | Update workspace name or description |
| **Workspaces** | DELETE | `/v1/workspaces/{workspace_id}` | Bearer JWT | Delete workspace and all documents |
| **Workspaces** | GET | `/v1/workspaces/public/default` | Optional JWT | Get or create default public workspace |
| **Workspaces** | POST | `/v1/workspaces/{workspace_id}/documents` | Bearer JWT | Upload and ingest document |
| **Workspaces** | GET | `/v1/workspaces/{workspace_id}/documents` | Bearer JWT | List documents in workspace |
| **Workspaces** | DELETE | `/v1/workspaces/{workspace_id}/documents/{doc_id}` | Bearer JWT | Delete document and vector chunks |
| **Documents** | GET | `/v1/citations/{citation_id}` | Bearer JWT | Retrieve citation metadata and excerpt |
| **Documents** | GET | `/v1/documents/{document_id}/viewer` | Bearer JWT | Get signed URL for browser document viewer |
| **Documents** | GET | `/v1/documents/{document_id}/highlights` | Bearer JWT | Get PDF bounding box coordinates |
| **Documents** | GET | `/v1/documents/files/{file_path:path}` | Signed Token | Stream document file content |
| **Documents** | POST | `/v1/documents/export/pdf` | Bearer JWT | Generate and download styled PDF report |
| **Documents** | GET | `/v1/documents/roic/transcript` | Bearer JWT | Search & retrieve Roic.ai earnings transcript |
| **GenAI Chat** | POST | `/v1/ai/chats` | Bearer JWT | Create new chat session |
| **GenAI Chat** | GET | `/v1/ai/chats` | Bearer JWT | List user chat sessions |
| **GenAI Chat** | GET | `/v1/ai/chats/{chat_id}` | Bearer JWT | Get chat session details |
| **GenAI Chat** | DELETE | `/v1/ai/chats/{chat_id}` | Bearer JWT | Delete chat session |
| **GenAI Chat** | GET | `/v1/ai/chats/public/default` | Optional JWT | Get public default chat session |
| **GenAI Chat** | GET | `/v1/ai/chats/{chat_id}/messages` | Bearer JWT | Get message history for chat |
| **GenAI Chat** | POST | `/v1/ai/chats/{chat_id}/messages` | Bearer JWT | Send user query & stream SSE AI response |
| **GenAI Chat** | GET | `/v1/ai/generations/active` | Bearer JWT | Get active in-flight generation status |
| **GenAI Chat** | GET | `/v1/ai/generations/{gen_id}/stream` | Bearer JWT | Reconnect to active SSE generation stream |
| **GenAI Chat** | POST | `/v1/ai/chats/{chat_id}/stop` | Bearer JWT | Stop active generation by chat ID |
| **GenAI Chat** | POST | `/v1/ai/generations/{gen_id}/stop` | Bearer JWT | Stop active generation by generation ID |
| **Takeaways** | POST | `/v1/workspaces/{ws_id}/takeaways` | Bearer JWT | Create multi-document analysis grid |
| **Takeaways** | GET | `/v1/workspaces/{ws_id}/takeaways` | Bearer JWT | List takeaway grids in workspace |
| **Takeaways** | GET | `/v1/workspaces/{ws_id}/takeaways/{grid_id}` | Bearer JWT | Get takeaway grid structure & values |
| **Takeaways** | POST | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/columns` | Bearer JWT | Add custom AI extraction column |
| **Takeaways** | DELETE | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/columns/{col_id}` | Bearer JWT | Delete grid column |
| **Takeaways** | POST | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/documents` | Bearer JWT | Add documents to existing grid |
| **Takeaways** | POST | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/cells/{cell_id}/retry` | Bearer JWT | Retry generation for a single cell |
| **Takeaways** | POST | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/duplicate` | Bearer JWT | Clone grid with column schema |
| **Takeaways** | PATCH | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/rename` | Bearer JWT | Rename takeaway grid |
| **Takeaways** | DELETE | `/v1/workspaces/{ws_id}/takeaways/{grid_id}` | Bearer JWT | Delete takeaway grid |
| **Takeaways** | GET | `/v1/workspaces/{ws_id}/takeaways/{grid_id}/export` | Bearer JWT | Export grid (CSV, JSON, Markdown) |
| **Takeaways** | GET | `/v1/document-takeaways` | Bearer JWT | List all takeaway grids across workspaces |
| **Skills** | GET | `/v1/skills/suggestions` | Bearer JWT | Get skill prompt suggestion chips |
| **Skills** | POST | `/v1/skills/generate-draft` | Bearer JWT | AI draft generation for a skill |
| **Skills** | POST | `/v1/skills/validate-draft` | Bearer JWT | Structural & quality validation of skill |
| **Skills** | POST | `/v1/skills` | Bearer JWT | Create new custom skill |
| **Skills** | GET | `/v1/skills` | Bearer JWT | List user's skills |
| **Skills** | GET | `/v1/skills/{skill_id}` | Bearer JWT | Get single skill details |
| **Skills** | PATCH | `/v1/skills/{skill_id}` | Bearer JWT | Update skill instructions or metadata |
| **Skills** | POST | `/v1/skills/{skill_id}/publish` | Bearer JWT | Publish skill for active use |
| **Skills** | DELETE | `/v1/skills/{skill_id}` | Bearer JWT | Delete custom skill |
| **Agents** | POST | `/v1/agents/validate` | Bearer JWT | Validate agent configuration & schedule |
| **Agents** | POST | `/v1/agents` | Bearer JWT | Create new Agent Workflow |
| **Agents** | GET | `/v1/agents` | Bearer JWT | List user's Agent Workflows |
| **Agents** | GET | `/v1/agents/scheduler/status` | Bearer JWT | Get background scheduler status |
| **Agents** | POST | `/v1/agents/scheduler/tick` | Bearer JWT | Trigger manual scheduler evaluation cycle |
| **Agents** | GET | `/v1/agents/tools` | Bearer JWT | List available tools for agent workflows |
| **Agents** | GET | `/v1/agents/templates` | Bearer JWT | List 10 landing page agent templates |
| **Agents** | POST | `/v1/agents/run` | Bearer JWT | Execute ad-hoc agent workflow |
| **Agents** | GET | `/v1/agents/runs/{run_id}` | Bearer JWT | Get run execution details & final output |
| **Agents** | GET | `/v1/agents/runs/{run_id}/steps` | Bearer JWT | Get step execution logs & traces |
| **Agents** | GET | `/v1/agents/{agent_id}` | Bearer JWT | Get Agent Workflow by ID |
| **Agents** | PATCH | `/v1/agents/{agent_id}` | Bearer JWT | Update Agent Workflow configuration |
| **Agents** | POST | `/v1/agents/{agent_id}/activate` | Bearer JWT | Activate Agent Workflow |
| **Agents** | POST | `/v1/agents/{agent_id}/pause` | Bearer JWT | Pause Agent Workflow |
| **Agents** | POST | `/v1/agents/{agent_id}/deactivate` | Bearer JWT | Deactivate Agent Workflow |
| **Agents** | DELETE | `/v1/agents/{agent_id}` | Bearer JWT | Delete Agent Workflow & history |
| **Agents** | POST | `/v1/agents/{agent_id}/run` | Bearer JWT | Trigger manual immediate run for agent |
| **Agents** | POST | `/v1/agents/{agent_id}/trigger-schedule` | Bearer JWT | Force scheduled execution for agent |
| **Agents** | GET | `/v1/agents/{agent_id}/runs` | Bearer JWT | List execution history runs for agent |
| **Prompt Templates** | GET | `/v1/cards/prompt-templates` | No | List active default landing prompt templates |

---

## Section F: Authentication & Authorization

### 1. Authentication Flow
CapitalEdge supports both native email/password authentication and Google OAuth 2.0.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Client as Web Frontend
    participant API as FastAPI Backend
    participant DB as PostgreSQL
    participant Mail as SMTP Service
    participant Google as Google Identity

    alt Native Registration
        User->>Client: Enter Email, Password, Name
        Client->>API: POST /v1/user/register
        API->>DB: Check uniqueness & Hash Password (Bcrypt)
        API->>DB: Generate 6-digit OTP in email_otps
        API->>Mail: Send verification email
        API-->>Client: Registration success (is_active=False)
        User->>Client: Enter OTP
        Client->>API: POST /v1/user/verify-otp
        API->>DB: Validate OTP & set is_active=True
        API-->>Client: Verification success
    else Native Login
        User->>Client: Enter Email & Password
        Client->>API: POST /v1/user/login
        API->>DB: Query User & Verify Bcrypt Hash
        API-->>Client: Return Access Token (14d) & Refresh Token (28d)
    else Google OAuth
        User->>Client: Sign in with Google
        Client->>Google: Authenticate & receive ID Token
        Client->>API: POST /v1/user/google/login (credential=ID Token)
        API->>Google: Verify ID token signature via google-auth
        API->>DB: Find or create User by google_id / email
        API-->>Client: Return Access Token (14d) & Refresh Token (28d)
    end
```

### 2. Token Specifications & Security
* **Access Token:** Stateless HMAC-SHA256 JWT signed with `AUTH_TOKEN`. Expiration: 20,160 minutes (14 days). Payload contains `{"sub": user_id, "exp": ...}`.
* **Refresh Token:** HMAC-SHA256 JWT signed with `AUTH_TOKEN`. Expiration: 40,320 minutes (28 days). Payload contains `{"sub": user_id, "type": "refresh", "exp": ...}`.
* **Password Hashing:** Passlib `CryptContext(schemes=["bcrypt"], deprecated="auto")`.
* **OTP Tokens:** 6-digit random integers stored in `email_otps` with `expires_at = utcnow() + 10 minutes` and `attempts` counter (maximum 5 tries).

### 3. Dependency Injection & Protected Routes
Protected routes declare `current_user: User = Depends(get_current_user)`. The dependency extracts the Bearer token from the `Authorization` header, decodes the JWT, queries the database for the active user, applies any due daily credit resets (`CreditService.check_and_apply_daily_reset`), and injects the user instance. Optional routes use `get_optional_current_user`, which returns `None` if no valid token is provided.

---

## Section G: Database Architecture & Schema

The database is built on **PostgreSQL 15+** with the **`pgvector`** extension enabled.

### 1. Entity Relationship (ER) Diagram

```mermaid
erDiagram
    users ||--o{ email_otps : "has"
    users ||--o{ help_queries : "submits"
    users ||--o{ credit_usage_logs : "incurs"
    users ||--o{ workspaces : "owns"
    users ||--o{ documents : "uploads"
    users ||--o{ chat_sessions : "creates"
    users ||--o{ skills : "authors"
    users ||--o{ agent_workflows : "configures"
    users ||--o{ agent_workflow_executions : "executes"

    workspaces ||--o{ documents : "contains"
    workspaces ||--o{ document_takeaway_grids : "hosts"
    workspaces ||--o{ chat_sessions : "scopes"

    documents ||--o{ document_chunks : "chunked_into"
    documents ||--o{ document_chat_links : "linked_to"
    chat_sessions ||--o{ document_chat_links : "includes"
    chat_sessions ||--o{ chat_messages : "contains"

    agent_workflows ||--o{ agent_workflow_executions : "runs"
    agent_workflow_executions ||--o{ agent_workflow_step_logs : "logs"

    document_takeaway_grids ||--o{ document_takeaway_grid_columns : "defines"
    document_takeaway_grids ||--o{ document_takeaway_grid_rows : "contains"
    document_takeaway_grid_rows ||--o{ document_takeaway_grid_cells : "has"
    document_takeaway_grid_columns ||--o{ document_takeaway_grid_cells : "populates"
```

---

### 2. Complete Database Models & Tables

#### A. Table `users`
* **Model:** `User` (`server/models/user.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `email` (VARCHAR(255), UNIQUE, NOT NULL, INDEXED)
  * `hashed_password` (VARCHAR(255), NULLABLE for Google-only users)
  * `full_name` (VARCHAR(255), NULLABLE)
  * `display_name` (VARCHAR(255), NULLABLE)
  * `avatar` (VARCHAR(500), NULLABLE)
  * `google_id` (VARCHAR(255), UNIQUE, NULLABLE)
  * `is_active` (BOOLEAN, DEFAULT FALSE)
  * `is_superuser` (BOOLEAN, DEFAULT FALSE)
  * `role` (VARCHAR(50), DEFAULT "member")
  * `subscription_tier` (VARCHAR(50), DEFAULT "free")
  * `subscription_status` (VARCHAR(50), DEFAULT "active")
  * `subscription_started_at` (DATETIME, NULLABLE)
  * `subscription_expires_at` (DATETIME, NULLABLE)
  * `daily_credits` (INTEGER, DEFAULT 100)
  * `credits_remaining` (INTEGER, DEFAULT 100)
  * `last_credit_reset_at` (DATETIME, DEFAULT utcnow)
  * `display_language` (VARCHAR(50), DEFAULT "English")
  * `nickname` (VARCHAR(100), NULLABLE)
  * `occupation` (VARCHAR(150), NULLABLE)
  * `bio` (TEXT, NULLABLE)
  * `ai_traits` (TEXT, DEFAULT '["Analytical", "Concise", "Data-Driven"]')
  * `custom_instructions` (TEXT, NULLABLE)
  * `ai_data_retention` (BOOLEAN, DEFAULT TRUE)
  * `settings_json` (TEXT, NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow, ON UPDATE utcnow)

#### B. Table `email_otps`
* **Model:** `EmailOtp` (`server/models/user.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `email` (VARCHAR(255), NOT NULL, INDEXED)
  * `otp` (VARCHAR(10), NOT NULL)
  * `purpose` (VARCHAR(50), DEFAULT "verification")
  * `attempts` (INTEGER, DEFAULT 0)
  * `is_used` (BOOLEAN, DEFAULT FALSE)
  * `expires_at` (DATETIME, NOT NULL)
  * `created_at` (DATETIME, DEFAULT utcnow)

#### C. Table `help_queries`
* **Model:** `HelpQuery` (`server/models/user.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE)
  * `email` (VARCHAR(255), NOT NULL)
  * `subject` (VARCHAR(255), NOT NULL)
  * `message` (TEXT, NOT NULL)
  * `status` (VARCHAR(50), DEFAULT "open")
  * `created_at` (DATETIME, DEFAULT utcnow)

#### D. Table `credit_usage_logs`
* **Model:** `CreditUsageLog` (`server/models/user.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `amount` (INTEGER, NOT NULL)
  * `action_type` (VARCHAR(100), NOT NULL)
  * `description` (TEXT, NULLABLE)
  * `balance_after` (INTEGER, NOT NULL)
  * `created_at` (DATETIME, DEFAULT utcnow, INDEXED)

#### E. Table `workspaces`
* **Model:** `Workspace` (`server/models/workspace.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `name` (VARCHAR(255), NOT NULL)
  * `description` (TEXT, NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow)

#### F. Table `documents`
* **Model:** `Document` (`server/models/workspace.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE CASCADE, INDEXED)
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `file_name` (VARCHAR(255), NOT NULL)
  * `file_format` (VARCHAR(50), NOT NULL)
  * `storage_path` (VARCHAR(500), NOT NULL)
  * `created_at` (DATETIME, DEFAULT utcnow)

#### G. Table `document_chunks`
* **Model:** `DocumentChunk` (`server/models/workspace.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `document_id` (VARCHAR, FK `documents.id` ON DELETE CASCADE, INDEXED)
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE CASCADE, INDEXED)
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `file_name` (VARCHAR(255), NULLABLE)
  * `chunk_number` (INTEGER, NOT NULL)
  * `total_chunks` (INTEGER, NOT NULL)
  * `content` (TEXT, NOT NULL)
  * `embedding` (VECTOR(1536), NOT NULL) — Indexed with HNSW cosine distance: `ix_document_chunks_embedding`
  * `start_offset` (INTEGER, NULLABLE)
  * `end_offset` (INTEGER, NULLABLE)
  * `page_number` (INTEGER, NULLABLE)
  * `section_title` (VARCHAR(255), NULLABLE)
  * `document_path` (VARCHAR(500), NULLABLE)
  * `document_url` (VARCHAR(500), NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)

#### H. Table `chat_sessions`
* **Model:** `ChatSession` (`server/models/genai.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE SET NULL, NULLABLE, INDEXED)
  * `title` (VARCHAR(255), NOT NULL)
  * `agent_id` (VARCHAR(255), NULLABLE)
  * `agent_name` (VARCHAR(255), NULLABLE)
  * `execution_type` (VARCHAR(50), NULLABLE)
  * `execution_date` (VARCHAR(50), NULLABLE)
  * `chat_metadata` (JSON, NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow)

#### I. Table `chat_messages`
* **Model:** `ChatMessage` (`server/models/genai.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `chat_id` (VARCHAR, FK `chat_sessions.id` ON DELETE CASCADE, INDEXED)
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE SET NULL, NULLABLE)
  * `role` (VARCHAR(50), NOT NULL) — "user" or "assistant"
  * `content` (TEXT, NOT NULL)
  * `document_ids` (JSON, NULLABLE)
  * `source_documents` (JSON, NULLABLE)
  * `citations` (JSON, NULLABLE)
  * `skills_used` (JSON, NULLABLE)
  * `agent_id` (VARCHAR(255), NULLABLE)
  * `agent_name` (VARCHAR(255), NULLABLE)
  * `mode_used` (VARCHAR(50), DEFAULT "auto")
  * `status` (VARCHAR(50), DEFAULT "complete")
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow)

#### J. Table `document_chat_links`
* **Model:** `DocumentChatLink` (`server/models/workspace.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `document_id` (VARCHAR, FK `documents.id` ON DELETE CASCADE)
  * `chat_id` (VARCHAR, FK `chat_sessions.id` ON DELETE CASCADE)
  * `created_at` (DATETIME, DEFAULT utcnow)
* **Constraint:** Unique `(document_id, chat_id)`.

#### K. Table `skills`
* **Model:** `Skill` (`server/models/skill.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `name` (VARCHAR(255), NOT NULL)
  * `description` (TEXT, NOT NULL)
  * `trigger_description` (TEXT, NOT NULL)
  * `instructions` (TEXT, NOT NULL)
  * `status` (VARCHAR(50), DEFAULT "draft") — "draft" or "published"
  * `version` (INTEGER, DEFAULT 1)
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow)
* **Constraint:** Unique `(user_id, name)`.

#### L. Table `agent_workflows`
* **Model:** `AgentWorkflow` (`server/models/agent_workflow.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE SET NULL, NULLABLE)
  * `name` (VARCHAR(255), NOT NULL)
  * `description` (TEXT, NULLABLE)
  * `instructions` (TEXT, NOT NULL)
  * `trigger_type` (VARCHAR(50), DEFAULT "automatic") — "automatic", "scheduled", "both"
  * `status` (VARCHAR(50), DEFAULT "draft") — "draft", "active", "paused", "completed"
  * `execution_mode` (VARCHAR(50), DEFAULT "auto") — "auto", "think_longer", "deep_research"
  * `limit_docs_past_7_days` (BOOLEAN, DEFAULT FALSE)
  * `selected_document_ids` (JSON, DEFAULT '[]')
  * `selected_sources` (JSON, DEFAULT '["workspace_documents", "sec_edgar"]')
  * `custom_sources` (JSON, DEFAULT '[]')
  * `schedule_frequency` (VARCHAR(50), NULLABLE) — "daily", "weekly", "monthly", "hourly", "minutes", "custom"
  * `schedule_interval` (INTEGER, NULLABLE)
  * `schedule_day` (VARCHAR(50), NULLABLE)
  * `schedule_time` (VARCHAR(50), NULLABLE)
  * `schedule_timezone` (VARCHAR(100), DEFAULT "UTC")
  * `schedule_cron` (VARCHAR(100), NULLABLE)
  * `scheduled_count_limit` (INTEGER, NULLABLE)
  * `recipient_email` (VARCHAR(255), NULLABLE)
  * `last_run_at` (DATETIME, NULLABLE)
  * `next_run_at` (DATETIME, NULLABLE)
  * `activated_at` (DATETIME, NULLABLE)
  * `is_system_template` (BOOLEAN, DEFAULT FALSE)
  * `created_at` (DATETIME, DEFAULT utcnow)
  * `updated_at` (DATETIME, DEFAULT utcnow)
* **Constraint:** Unique `(user_id, name)`.

#### M. Table `agent_workflow_executions`
* **Model:** `AgentWorkflowExecution` (`server/models/agent_workflow.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `workflow_id` (VARCHAR, FK `agent_workflows.id` ON DELETE SET NULL, NULLABLE, INDEXED)
  * `user_id` (VARCHAR, FK `users.id` ON DELETE CASCADE, INDEXED)
  * `workspace_id` (VARCHAR, FK `workspaces.id` ON DELETE SET NULL, NULLABLE)
  * `trigger_source` (VARCHAR(50), NOT NULL) — "manual", "scheduled", "automatic"
  * `status` (VARCHAR(50), DEFAULT "pending") — "pending", "running", "completed", "failed", "cancelled"
  * `query_or_goal` (TEXT, NOT NULL)
  * `plan_steps` (JSON, DEFAULT '[]')
  * `final_response` (TEXT, NULLABLE)
  * `citations` (JSON, DEFAULT '[]')
  * `source_documents` (JSON, DEFAULT '[]')
  * `skills_applied` (JSON, DEFAULT '[]')
  * `tools_called` (JSON, DEFAULT '[]')
  * `shared_state` (JSON, DEFAULT '{}')
  * `error_message` (TEXT, NULLABLE)
  * `execution_time_ms` (INTEGER, NULLABLE)
  * `started_at` (DATETIME, DEFAULT utcnow)
  * `completed_at` (DATETIME, NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)

#### N. Table `agent_workflow_step_logs`
* **Model:** `AgentWorkflowStepLog` (`server/models/agent_workflow.py`)
* **Primary Key:** `id` (VARCHAR, UUID string)
* **Fields:**
  * `execution_id` (VARCHAR, FK `agent_workflow_executions.id` ON DELETE CASCADE, INDEXED)
  * `step_number` (INTEGER, NOT NULL)
  * `step_name` (VARCHAR(255), NOT NULL)
  * `status` (VARCHAR(50), DEFAULT "pending") — "pending", "running", "completed", "failed", "retrying"
  * `skill_name` (VARCHAR(255), NULLABLE)
  * `tool_name` (VARCHAR(255), NULLABLE)
  * `input_context` (JSON, NULLABLE)
  * `output_result` (JSON, NULLABLE)
  * `error_message` (TEXT, NULLABLE)
  * `retry_count` (INTEGER, DEFAULT 0)
  * `latency_ms` (INTEGER, NULLABLE)
  * `created_at` (DATETIME, DEFAULT utcnow)

#### O. Tables `document_takeaway_grids`, `columns`, `rows`, `cells`
* **Models:** `DocumentTakeawayGrid`, `DocumentTakeawayGridColumn`, `DocumentTakeawayGridRow`, `DocumentTakeawayGridCell` (`server/models/document_takeaways.py`)
* **Grid Fields:** `id`, `workspace_id`, `name`, `status`, `created_at`, `updated_at`.
* **Column Fields:** `id`, `grid_id`, `name`, `column_type` ("document" or "ai_generated"), `instruction` (prompt), `display_order`.
* **Row Fields:** `id`, `grid_id`, `document_id` (FK `documents.id`), `display_order`.
* **Cell Fields:** `id`, `row_id`, `column_id`, `value` (TEXT), `status` ("pending", "processing", "completed", "failed"), `error_message`, `updated_at`.

---

## Section H: Background Jobs, Scheduling & Concurrency

### 1. The Asynchronous Agent Scheduler (`AgentSchedulerService`)
* **Lifespan Integration:** Initialized on application startup in `server/main.py` via `agent_scheduler_service.start(poll_interval_seconds=30)` and stopped cleanly on shutdown.
* **Candidate Discovery:** Queries all active agents with `trigger_type` in `["scheduled", "both"]`.
* **Cron & Interval Evaluation:** Evaluates 5-part cron expressions (`matches_cron`) or frequency rules (`daily`, `weekly`, `monthly`, `hourly`, `minutes`) against the agent's configured timezone.
* **Concurrency Locking:** Employs an in-memory `asyncio.Lock()` and a running set (`_running_agent_ids`) to prevent overlapping executions of the same agent.
* **Quota Enforcement:** Auto-pauses agents that have reached their `scheduled_count_limit`.
* **Execution & Notification:** Executes `AgentExecutionService` in an isolated database session, creates or appends to a daily chat session, generates a summary title via `AgentTitleService`, and dispatches a styled HTML report to `recipient_email` using `fastapi-mail`.
* **Stale Run Recovery:** On reboot, `recover_stale_executions(db)` finds any executions in `running` or `pending` state and transitions them to `failed` with `"Server restarted while task was in progress"`.

### 2. In-Flight Generation Runtime & Cancellation (`generation_runtime.py`)
* **The Disconnect Problem:** In standard SSE, if a user refreshes or disconnects, Starlette terminates the generator and drops unpersisted tokens.
* **The Solution (Decoupled Worker & Pub/Sub):**
  1. The chat endpoint registers a `GenerationHandle` in `generation_runtime.py`.
  2. The LLM producer runs on a background worker thread and pushes tokens to an in-memory queue.
  3. Starlette pumps a relay generator that pulls from the queue.
  4. If the client disconnects, only the relay generator closes; the producer completes and persists the message to PostgreSQL.
  5. If the client reconnects via `/v1/ai/generations/{id}/stream`, it receives all buffered tokens from offset 0 without re-running the LLM.
  6. If the user presses the **Stop** button (`/stop`), `handle.cancel()` sets a cancellation event. Pre-generation stages (rewriting, intent routing, retrieval) and the token loop check `raise_if_cancelled()`, immediately terminating execution and persisting partial text.

### 3. Asynchronous Grid Matrix Processing (`BackgroundTasks`)
* Grid creation queues `process_grid_cell_task` into FastAPI `BackgroundTasks`.
* Cells process concurrently, fetching document chunks via RAG and updating cell statuses in PostgreSQL (`pending` -> `processing` -> `completed`).

---

## Section I: AI, LLM, RAG & Multi-Agent Engine

### 1. Models & Providers
* **Primary LLM (`settings.LLM_MODEL`):** `openai/gpt-4.1-mini` (via OpenRouter or direct OpenAI) for query rewriting, intent classification, planning, step reasoning, and RAG synthesis.
* **Reasoning / Think Longer Model (`settings.THINK_LONGER_MODEL`):** `openai/gpt-5.6-terra` with `reasoning_effort: medium/high`.
* **Deep Research Model (`settings.DEEP_RESEARCH_MODEL`):** `openai/gpt-5.6-terra`.
* **Embedding Model (`settings.EMBEDDING_MODEL`):** `text-embedding-3-small` (1536 dimensions).
* **Re-Ranking Model (`settings.RERANK_MODEL`):** `openai/gpt-4.1-mini`.

### 2. Two-Stage RAG Pipeline
```mermaid
flowchart LR
    Q[User Query] --> Embed[OpenAI Embedding text-embedding-3-small]
    Embed --> Stage1[Stage 1: pgvector Cosine Search HNSW Index]
    Stage1 --> Pool[Candidate Pool Top 20 Chunks]
    Pool --> Stage2[Stage 2: LLM Listwise Reranker RerankService]
    Stage2 --> TopK[Top K Context Chunks]
    TopK --> LLM[LLM Synthesis & Citation Generation]
```
1. **Stage 1 (Vector Search):** pgvector calculates cosine distance `embedding.cosine_distance(query_embedding)` using the HNSW index (`ef_search = 40`).
2. **Stage 2 (Cross-Encoder Re-Ranking):** `RerankService` presents the query and candidate passages to `gpt-4.1-mini`, asking for a listwise ranking `[3, 1, 2]`. On failure, it gracefully degrades to vector order.
3. **Context Sufficiency Check (DEF-011):** Checks if retrieved document chunks adequately answer the prompt. If insufficient, automatically triggers live web search fallback.
4. **Finance MCP Sufficiency Check (DEF-012):** Evaluates if SEC/Roic/News data is complete. If sparse, supplements with live web search.

### 3. Autonomous Multi-Agent Cognitive Loop (`AgentWorkflowEngine`)
1. **Receive Request & Initialize:** Creates `AgentSharedState` blackboard and records `AgentWorkflowExecution` (status `pending`).
2. **Task Decomposition (Planning):** Prompts LLM to break goal into 1-4 atomic steps with `tool_hint` and `skill_hint`.
3. **Step Execution Loop:**
   * Selects next step and resolves applicable skill instructions via `SkillSelectionService`.
   * Prepares tool context (`rag`, `sec_edgar`, `roic_ai`, `financial_news`, `companies_house`, `web_search`).
   * Executes tool and feeds output to LLM step reasoner.
   * On error, executes exponential backoff retry (up to 2 retries).
   * Updates `AgentSharedState` blackboard with chunks, facts, and variables.
   * Logs step latency and status in `agent_workflow_step_logs`.
4. **Final Synthesis:** Synthesizes accumulated step findings into an executive report with citations, source attributions, and follow-up questions.

---

## Section J: File & Document Processing Pipeline

```mermaid
flowchart TD
    Upload[Uploaded File] --> Validate{Type & 10MB Check}
    Validate -- Valid --> SaveDisk[Save to static/workspace_docs/UUID.ext]
    SaveDisk --> Extract{Native Text Available?}
    Extract -- Yes (PDF/DOCX/TXT/XLSX) --> ParseNative[PyMuPDF / docx / pandas]
    Extract -- No / Scanned Images --> OCR[OpenCV Preprocessing + RapidOCR/Tesseract]
    ParseNative --> Clean[Text Normalization clean_text]
    OCR --> Clean
    Clean --> Chunk[ChunkingService 1000 char window, 200 overlap]
    Chunk --> Embed[EmbeddingService OpenAI Batched Embeddings]
    Embed --> Persist[Store Document & DocumentChunks in PostgreSQL]
```

### Supported Formats & Extraction Engines
* **PDF (`.pdf`):** PyMuPDF (`fitz`) extracts native text; scanned pages are rasterized to pixmaps and processed via OCR.
* **Word (`.docx`, `.doc`):** `python-docx` extracts paragraphs and tables; `textract` fallback for legacy `.doc`.
* **Excel (`.xlsx`, `.xls`):** `openpyxl` / `pandas` / `xlrd` formats tabular data as markdown tables.
* **CSV (`.csv`):** `pandas` parses rows and columns.
* **Plain Text & Markdown (`.txt`, `.md`):** UTF-8 / Latin-1 decoding.
* **Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`):** OpenCV preprocessing (grayscale, bilateral filter denoising, Otsu thresholding) followed by RapidOCR (ONNX runtime) or Tesseract OCR.

---

## Section K: External Integrations & Model Context Protocol (MCP)

| Integration | File / Module | Purpose | Authentication / Config |
|---|---|---|---|
| **OpenAI / OpenRouter** | `server/utils/llm_client.py` | Embeddings, Chat, Planning, Reranking | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| **SerpAPI** | `server/services/rag_service.py` | Real-time live Google web search fallback | `SERPAPI_KEY` |
| **SEC EDGAR API** | `server/services/sec_edgar_mcp.py` | Official 10-K, 10-Q, 8-K filings & facts | `SEC_EDGAR_USER_AGENT` (User-Agent header required by SEC) |
| **Roic.ai API** | `server/services/roic_ai_mcp.py` | Financial ratios, valuation metrics, transcripts | `ROIC_API_KEY` |
| **UK Companies House** | `server/services/companies_house_mcp.py` | UK official company registry & officer profiles | `COMPANIES_HOUSE_API_KEY` |
| **Alpha Vantage** | `server/services/financial_news_mcp.py` | Market news & sentiment analysis | `ALPHA_VANTAGE_API_KEY` |
| **Google OAuth 2.0** | `server/api_v1/endpoints/user.py` | Google single sign-on authentication | `GOOGLE_CLIENT_ID` via `google-auth` |
| **FastAPI-Mail (SMTP)** | `server/utils/mail.py` | OTP emails, support tickets, scheduled reports | `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_SERVER`, `MAIL_PORT` |
| **FastMCP Server** | `server/mcp_server.py` | Standalone FastMCP server over stdio/SSE | Standardized FastMCP tools |

---

## Section L: Environment Variables Reference

| Variable Name | Type | Default Value | Description | Required |
|---|---|---|---|---|
| `PROJECT_NAME` | string | `CapitalEdge AI Enterprise` | Name of the application | Optional |
| `API_V1_STR` | string | `/v1` | URL prefix for version 1 API | Optional |
| `SQLALCHEMY_DATABASE_URI` | string | `postgresql://user:pass@localhost:5432/agentic_db` | PostgreSQL connection string | **Yes** |
| `AUTH_TOKEN` | string | `capitaledge-enterprise-secret-key-change-in-production` | Secret key for JWT signing (HS256) | **Yes** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | integer | `20160` (14 days) | JWT access token lifetime | Optional |
| `REFRESH_TOKEN_EXPIRE_MINUTES`| integer | `40320` (28 days) | JWT refresh token lifetime | Optional |
| `OPENAI_API_KEY` | string | `""` | OpenAI or OpenRouter API key | **Yes** |
| `OPENAI_BASE_URL` | string | `https://openrouter.ai/api/v1` | Base URL for LLM and Embeddings | Optional |
| `LLM_MODEL` | string | `openai/gpt-4.1-mini` | Default chat and reasoning model | Optional |
| `THINK_LONGER_MODEL` | string | `openai/gpt-5.6-terra` | Think Longer reasoning model | Optional |
| `THINK_LONGER_REASONING_EFFORT`| string | `medium` | Reasoning effort (`low`, `medium`, `high`) | Optional |
| `DEEP_RESEARCH_MODEL` | string | `openai/gpt-5.6-terra` | Deep Research planning model | Optional |
| `DEEP_RESEARCH_REASONING_EFFORT`| string | `medium` | Deep Research reasoning effort | Optional |
| `EMBEDDING_MODEL` | string | `text-embedding-3-small` | OpenAI vector embedding model | Optional |
| `EMBEDDING_DIMENSIONS` | integer | `1536` | Vector embedding dimension size | Optional |
| `EMBEDDING_BATCH_SIZE` | integer | `100` | Chunk batch size for embedding calls | Optional |
| `RERANK_MODEL` | string | `openai/gpt-4.1-mini` | Cross-encoder listwise re-ranker model | Optional |
| `RAG_TOP_K` | integer | `8` | Top chunks returned after re-ranking | Optional |
| `RAG_RERANK_ENABLED` | boolean | `true` | Enable LLM second-stage re-ranking | Optional |
| `CHUNK_SIZE` | integer | `1000` | Text chunk character window size | Optional |
| `CHUNK_OVERLAP` | integer | `200` | Overlapping characters between chunks | Optional |
| `OCR_ENABLED` | boolean | `true` | Enable OCR on uploaded images/scans | Optional |
| `OCR_ENGINE` | string | `rapidocr` | OCR engine (`rapidocr`, `tesseract`, `easyocr`) | Optional |
| `SERPAPI_KEY` | string | `""` | SerpAPI key for Google web search | Optional |
| `SEC_EDGAR_USER_AGENT` | string | `CapitalEdge Research admin@capitaledge.ai` | User-Agent required by SEC EDGAR API | **Yes** |
| `ROIC_API_KEY` | string | `""` | Roic.ai financial ratios API key | Optional |
| `COMPANIES_HOUSE_API_KEY` | string | `""` | UK Companies House API key | Optional |
| `ALPHA_VANTAGE_API_KEY` | string | `""` | Alpha Vantage financial news API key | Optional |
| `MAIL_USERNAME` | string | `""` | SMTP username | Optional |
| `MAIL_PASSWORD` | string | `""` | SMTP password | Optional |
| `MAIL_FROM` | string | `support@capitaledge.ai` | From email address for notifications | Optional |
| `MAIL_PORT` | integer | `587` | SMTP server port | Optional |
| `MAIL_SERVER` | string | `smtp.gmail.com` | SMTP server hostname | Optional |
| `PDF_EXPORT_COMPANY_NAME` | string | `CapitalEdge` | Brand name printed on PDF exports | Optional |

---

## Section M: Error Handling, Auditing & Observability

### 1. Global Exception Handlers (`server/main.py`)
* `StarletteHTTPException`: Returns structured JSON `{"success": false, "error": {"code": status, "message": detail}}`.
* `RequestValidationError`: Formats Pydantic schema validation failures with exact field paths and error messages.
* `Exception` (Unhandled 500): Logs full stack trace to `logs/audit.log` and returns `{"success": false, "error": {"code": 500, "message": "Internal server error"}}`.

### 2. Structured JSON Audit Logging (`server/services/audit_logger.py`)
All operations write structured JSON records to `logs/audit.log` and console:
```json
{
  "timestamp": "2026-09-03T10:00:00.000Z",
  "action": "CHAT_ORCHESTRATION",
  "user_id": "u-12345",
  "workspace_id": "ws-987",
  "chat_id": "chat-456",
  "endpoint": "/v1/ai/chats/chat-456/messages",
  "http_method": "POST",
  "status": "success",
  "processing_time_ms": 1420,
  "metadata": {
    "providers": ["RAG", "SEC EDGAR"],
    "rewritten_query": "Microsoft cloud growth FY25",
    "citations_count": 3
  }
}
```

---

## Section N: Production Architecture & Deployment

### 1. Runtime Stack
* **Python Runtime:** Python 3.10+ (Docker image uses `python:3.10-slim`, tested on Python 3.14).
* **ASGI Server:** Uvicorn (single-worker in development, multiple workers behind Gunicorn in production).
* **Process Manager:** PM2 (`ecosystem.config.js`) or Docker Compose.
* **Reverse Proxy:** Nginx with SSL termination, HTTP/2, and SSE buffering disabled (`proxy_buffering off;`).
* **Database:** PostgreSQL 15+ with `pgvector` extension.

### 2. Automated Linux / EC2 Server Setup (`serversetup.sh`)
The repository includes a production setup script that automates system dependencies:
```bash
bash serversetup.sh
```
* Installs system packages: `python3-venv`, `libpq-dev`, `tesseract-ocr`, `libgl1-mesa-glx`, `build-essential`.
* Creates virtual environment `./venv`.
* Installs `requirements.txt`.
* Generates storage directories (`static/workspace_docs`, `static/avatars`, `logs`).

### 3. Database Migration Execution
```bash
# Run migrations using Alembic
alembic upgrade head
```

### 4. Docker Deployment
```bash
# Build and run multi-container stack (App + PostgreSQL + pgvector)
docker compose up -d --build
```

---

## Section O: Troubleshooting & Operational Runbook

| Problem | Root Cause | Step-by-Step Resolution |
|---|---|---|
| `type "vector" does not exist` | `pgvector` extension not enabled in PostgreSQL | Connect to Postgres as superuser: `CREATE EXTENSION IF NOT EXISTS vector;` then re-run `alembic upgrade head`. |
| `ModuleNotFoundError: No module named 'fastapi'` | Virtual environment not active or packages missing | Run `source venv/bin/activate && pip install -r requirements.txt`. Ensure scripts run with `PYTHONPATH=.`. |
| `TesseractNotFoundError` | Tesseract binary missing while using `OCR_ENGINE="tesseract"` | Set `OCR_ENGINE=rapidocr` in `.env` (pure Python ONNX, no OS binary needed) or install Tesseract: `sudo apt-get install -y tesseract-ocr`. |
| Streaming SSE connections drop unexpectedly | Nginx is buffering SSE responses | Add `proxy_buffering off; proxy_cache off; proxy_read_timeout 300s;` to the Nginx location block. |
| OpenAI API Rate Limit (429) or Timeout | OpenAI API quota exhausted or transient network lag | The backend automatically applies exponential retries (up to 3 attempts). Verify `OPENAI_API_KEY` and consider using OpenRouter endpoints. |
| Scheduled Agents not executing | Background scheduler stopped or timezone mismatch | Check `GET /v1/agents/scheduler/status`. Verify `schedule_timezone` matches agent local time. Trigger manual tick with `POST /v1/agents/scheduler/tick`. |
| `stale_executions` on startup | Server was restarted while background agents were running | The backend automatically marks orphaned runs as `failed` on startup via `recover_stale_executions()`. Check `logs/audit.log`. |
| Credit consumption race condition | Simultaneous requests for the same user | `CreditService` utilizes `crud.user.get_with_lock()` with PostgreSQL row-level locks (`with_for_update`) to ensure atomic balance checks. |
