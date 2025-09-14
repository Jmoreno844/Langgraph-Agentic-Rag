# langgraph_agentic_rag

A production-minded RAG chatbot backend built with LangGraph + FastAPI. It ingests documents from S3, synchronizes them with a Pinecone vector store, and serves a streaming chat endpoint that uses a sophisticated hybrid search and reranking retrieval strategy.

## Features

- RAG graph with conditional tool routing
- S3-backed document management API that syncs with Pinecone
- Hybrid search (Pinecone vector search + BM25) with VoyageAI reranking for retrieval
- FastAPI app with streaming `/chat` endpoint
- Async-first design with `AsyncPostgresSaver` for persistent, stateful conversations
- Observability with Langsmith tracing
- RAG evaluation suite using DeepEval
- Pytest unit tests for key nodes

## Quickstart

1.  Prereqs

    - Python 3.12
    - An S3 bucket with documents to ingest
    - API keys: OpenAI, AWS, VoyageAI
    - A running Postgres database

2.  Install

    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  Configure environment

    - Create a `.env` file at the project root (see `docs/CONFIG.md` for details).
    - Minimum required for local run:

    ```bash
    # Core services
    OPENAI_API_KEY=...
    VOYAGE_API_KEY=...
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_S3_RAG_DOCUMENTS_BUCKET=your-bucket
    AWS_DB_URL=postgresql+psycopg://user:pass@host:5432/dbname

    # Observability (recommended)
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=...
    LANGCHAIN_PROJECT=your-langsmith-project-name
    ```

4.  Run the API

    ```bash
    # Kill process on port 8000 if it's running
    lsof -ti :8000 | xargs -r kill -9
    # Start the server
    uvicorn src.app.main:app --reload
    ```

    - Open API docs at http://127.0.0.1:8000/docs

## API Overview

- GET `/` → health check
- POST `/chat` → streaming Server-Sent Events (SSE) with model output

  - Request body example:

  ```json
  {
    "session_id": "demo-session-1",
    "messages": [{ "role": "user", "content": "What is the warranty?" }]
  }
  ```

- Documents API (S3-backed)
  - GET `/documents` → list objects (optionally with presigned URLs)
  - POST `/documents` → upload
  - PUT `/documents/{key}` → update/replace
  - DELETE `/documents/{key}` → delete

## Testing & Evaluation

- **Unit Tests**: Fast, mocked tests for individual components.
  ```bash
  pytest -q
  ```
- **RAG Evaluation**: Quality and performance metrics using DeepEval.
  - See `evals/README.md` for setup and execution details.

## Project Docs

- Architecture: `docs/ARCHITECTURE.md`
- Configuration: `docs/CONFIG.md`
- Testing guide: `docs/TESTING.md`
- Runbook (common issues): `docs/RUNBOOK.md`

## Notes

- The RAG retriever uses Pinecone for hybrid search (dense vectors + sparse BM25 keywords) and VoyageAI for reranking to provide highly relevant context.
- The `/documents` API automatically syncs uploaded S3 documents with the configured Pinecone index.
