# Runbook

Minimal guidance for common local issues.

## API does not start

- **Symptom**: Crash on startup with a database error.
- **Check**: `AWS_DB_URL` is set to an `asyncpg`-compatible URI (`postgresql+psycopg://...`) and is reachable. Example in `docs/CONFIG.md`.
- **Check**: Network access/security rules to the database.
- **Symptom**: `Address already in use`.
- **Fix**: A previous server process is still running. Find and kill it.
  ```bash
  # Find process ID (PID) on port 8000 and kill it
  lsof -ti :8000 | xargs -r kill -9
  ```

## S3 or Pinecone errors

- **Symptom**: Errors from S3 or Pinecone during document upload or retrieval.
- **Check env**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_RAG_DOCUMENTS_BUCKET`.
- **Check env**: `PINECONE_API_KEY`, `PINECONE_INDEX`.
- **Check**: Ensure the S3 bucket and Pinecone index exist and are configured correctly.

## Traces not appearing in Langsmith

- **Symptom**: The app runs, but no traces are logged to your Langsmith project.
- **Check env**:
  - `LANGCHAIN_TRACING_V2` must be set to `"true"`.
  - `LANGCHAIN_API_KEY` must be a valid key.
  - `LANGCHAIN_PROJECT` must match the project name in Langsmith.

## Missing API keys

- **Symptom**: Model/embedding calls fail with authentication errors.
- **Check env**: `OPENAI_API_KEY`, `VOYAGE_API_KEY`.

## Quick health checks

- **Start API**:
  ```bash
  lsof -ti :8000 | xargs -r kill -9
  uvicorn src.app.main:app --reload
  ```
- **Open docs**: http://127.0.0.1:8000/docs
- **Health**: `curl http://127.0.0.1:8000/`
- **Chat (SSE test)**:
  ```bash
  curl -N --no-progress-meter --http1.1 \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -X POST http://127.0.0.1:8000/chat \
    -d '{
          "session_id": "local-1",
          "messages": [{"role": "user", "content": "What is the warranty?"}]
        }'
  ```
