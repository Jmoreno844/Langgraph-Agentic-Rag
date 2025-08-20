# Testing guidelines

This test suite uses pytest and focuses on fast, deterministic unit tests. External services (LLMs, vector DBs, network I/O) are mocked.

## How to run

```bash
pytest -q
```

## Principles

- Keep tests deterministic: mock network, LLMs, and time-dependent behaviors
- Test behavior over implementation details: assert inputs/outputs and routing decisions
- Prefer fixtures for shared setup; avoid global state
- Small, focused tests over large, brittle ones
- Use monkeypatch to replace external dependencies

## Structure

- `tests/graph/routing/`: routers and conditional edges
- `tests/graph/nodes/`: node-level behavior
- `tests/graph/vectorstores/`: retrievers and tools

## Environment

- Do not rely on real API keys for tests. LLM calls must be monkeypatched to fake models
- Avoid real databases. Use in-memory or mock checkpointers for integration tests

## Adding new tests

- Name tests descriptively: `test_<unit>_<expected_behavior>()`
- Assert both the outcome and key side effects (e.g., messages appended, flags set)
- When mocking models, capture prompts to validate prompt engineering

## Test modules and purpose

- **tests/conftest.py**

  - Purpose:
    - Ensure repository root is importable via `sys.path` so `import src.*` works in tests.
    - Set safe default environment variables for hermetic test runs without real API keys.

- **tests/graph/nodes/test_generate_answer.py**

  - Purpose:
    - Validate prompt composition includes both the question (from `messages[-3]`) and retrieved context (from `messages[-1]`).
    - Verify basic response shape returns an `AIMessage` object.
    - Test error handling when state has insufficient messages (underspecified state raises an exception).

- **tests/graph/nodes/test_generate_answer_or_rag.py**

  - Purpose:
    - `'Not rewritten'` path: inject a `SystemMessage` when `has_been_rewritten` is `False`, bind the retriever tool, and keep the flag as `False`.
    - `'Already rewritten'` path: do not inject `SystemMessage` when `has_been_rewritten` is `True`, preserve the flag, and pass through tool-call responses correctly.
    - Verify tools are bound for potential routing decisions in both scenarios.

- **tests/graph/nodes/test_rewrite_question.py**

  - Purpose:
    - Verify the rewrite prompt is formed from the user's original question (extracted from `messages[-2]`).
    - Ensure returned state contains a rewritten `AIMessage` and sets `has_been_rewritten=True`.
    - Validate prompt content includes the original question text for LLM processing.

- **tests/graph/routing/test_document_grader.py**

  - Purpose:
    - `'Bypass when already rewritten'`: when `has_been_rewritten=True`, route directly to `generate_answer` without calling the grader model.
    - `'Route on relevant'`: when grader returns "yes" score, route to `generate_answer`.
    - `'Route on irrelevant'`: when grader returns "no" score, route to `rewrite_question`.
    - Sanity-check that the grading prompt includes both question and context strings for proper evaluation.

- **tests/graph/vectorstores/test_in_memory.py**

  - Purpose:
    - `'Error when no documents'`: raise a clear `ValueError` when document loader returns an empty list.
    - `'Happy path with stubs'`: stub external dependencies (embeddings, vector store, reranker); create a retriever tool with expected name (`retrieve_rag_docs`) and description; verify retriever wiring returns documents via `invoke`.
    - Keep tests hermetic by pre-stubbing import-time modules (`langchain_voyageai`, `local_loader_splitter`).

- **tests/graph/orchestration/test_graph.py**

  - Purpose:
    - Compile the complete graph with `MemorySaver` checkpointer and faked LLM/retriever dependencies.
    - `'Direct answer path'`: when LLM returns a plain response (no tool calls), verify final state produces an `AIMessage`.
    - `'RAG path'`: when LLM requests tool usage (tool-call → retriever → `generate_answer`), verify the complete flow produces a final `AIMessage`.

- **tests/api/test_app.py**
  - Purpose:
    - `'Root endpoint'`: verify `GET /` returns expected Hello World JSON payload with 200 status.
    - `'Chat streaming contract'`: verify `POST /chat` streams Server-Sent Events with proper start/data/end event structure using a compiled in-memory graph (with fake LLM and stub retriever).
