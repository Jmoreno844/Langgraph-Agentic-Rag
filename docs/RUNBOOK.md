# Runbook

Minimal guidance for common local issues.

## API does not start

- Symptom: crash on startup with a database error
- Check: `AWS_DB_URL` is set and reachable (Postgres). Example connection string in `docs/CONFIG.md`.
- Check: Network access/security rules to the database.

## S3 errors or empty retrieval

- Symptom: errors from S3 or `ValueError("No documents loaded from S3.")`
- Check env: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_RAG_DOCUMENTS_BUCKET`
- Ensure the bucket exists and contains supported documents.
- Region is hard-coded as `us-east-1` in `src/services/s3_service.py`.

## Missing API keys

- Symptom: model/embedding calls fail
- Check env: `OPENAI_API_KEY`, `VOYAGE_API_KEY`

## Quick health checks

- Start API: `uvicorn src.app.main:app --reload`
- Open docs: http://127.0.0.1:8000/docs
- Health: `curl http://127.0.0.1:8000/`
- Chat (example payload):

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "session_id": "local-1",
        "messages": [{"role": "user", "content": "What is the warranty?"}]
      }'
```
