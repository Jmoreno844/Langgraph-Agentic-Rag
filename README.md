# langgraph_agentic_rag

A production-minded RAG chatbot backend built with LangGraph + FastAPI. It ingests documents from S3, chunks them, builds an in-memory retriever with reranking, and serves a streaming chat endpoint. Designed to showcase good software engineering practices.

## Features

- RAG graph with conditional tool routing
- S3-backed ingestion and document management API
- In-memory vector store + VoyageAI reranking for retrieval
- FastAPI app with streaming `/chat` endpoint
- Postgres-backed LangGraph checkpointer
- Pytest unit tests for key nodes

## Quickstart

1. Prereqs

- Python 3.12
- An S3 bucket with documents to ingest
- API keys: OpenAI, AWS, VoyageAI

2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3. Configure environment

- Create a `.env` file at the project root (see `docs/CONFIG.md` for details).
- Minimum required for local run:

```bash
OPENAI_API_KEY=...
VOYAGE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_RAG_DOCUMENTS_BUCKET=your-bucket
AWS_DB_URL=postgresql://user:pass@host:5432/dbname
```

4. Run the API

```bash
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

## Testing

```bash
pytest -q
```

- Unit tests live under `tests/graph/nodes/` and use fakes to avoid external services.

## Project Docs

- Architecture: `docs/ARCHITECTURE.md`
- Configuration: `docs/CONFIG.md`
- Testing guide: `docs/TESTING.md`
- Runbook (common issues): `docs/RUNBOOK.md`

## Notes

- The retriever currently uses an in-memory vector store with `OpenAIEmbeddings` and VoyageAI reranking.
- A Pinecone-based retriever exists in the codebase but is not wired by default. See `src/graph/vectorstores/` if needed.
