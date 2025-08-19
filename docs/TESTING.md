# Testing

This project includes unit tests that validate the behavior of graph nodes without calling external services.

## Run tests

```bash
pytest -q
```

## What is covered

- `tests/graph/nodes/test_generate_answer.py`
  - Prompt composition includes the question and retrieved context
  - Basic response shape
- `tests/graph/nodes/test_generate_answer_or_rag.py`
  - System message injection when not rewritten
  - Tool-call preservation and rewritten flag handling

## Patterns used

- **Fake LLMs**: Tests stub the response model to avoid network and cost, and to assert exact messages passed into the model.
- **No external services**: S3, Postgres, and LLM providers are avoided in unit tests for speed and determinism.

## Tips

- Run a single test file: `pytest tests/graph/nodes/test_generate_answer.py -q`
- Run a single test: `pytest tests/graph/nodes/test_generate_answer.py::test_generate_answer_formats_prompt_with_question_and_context -q`

> Integration tests and cloud-backed checks can be added later (e.g., Moto for S3, Testcontainers for Postgres) if desired. For now, unit tests emphasize correctness of node-level logic.
