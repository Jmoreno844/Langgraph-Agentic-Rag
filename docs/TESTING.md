# Testing guidelines

This project uses two distinct testing strategies: fast, isolated **unit tests** with `pytest` and comprehensive **RAG evaluation** with `DeepEval`.

## Unit Testing (`pytest`)

Unit tests focus on fast, deterministic checks of individual components (nodes, routers, etc.). External services like LLMs, databases, and network I/O are mocked to ensure tests are hermetic and quick to run.

### How to run

```bash
pytest -q
```

### Principles

- **Deterministic**: Mock network, LLMs, and time-dependent behaviors.
- **Behavior-driven**: Test inputs/outputs and routing decisions, not implementation details.
- **Focused**: Prefer small, targeted tests over large, brittle ones.
- **No external dependencies**: Do not rely on real API keys or databases. Use fakes, mocks, and in-memory stand-ins (`MemorySaver`).

## RAG Evaluation (`DeepEval`)

RAG evaluation focuses on the quality and performance of the entire system. It uses a synthetic dataset to measure the RAG Triad (Faithfulness, Answer Relevancy, Contextual Recall) and other key metrics.

- **For detailed setup, execution, and troubleshooting, see `evals/README.md`**.

## Test modules and purpose

- **`tests/conftest.py`**

  - **Purpose**: Ensures the `src` directory is importable in tests and sets safe default environment variables to prevent accidental calls to real services.

- **`tests/graph/nodes/test_generate_answer.py`**

  - **Purpose**: Validates that the final answer generation prompt correctly includes both the user's question and the retrieved context, ensuring the model has the necessary information to form a grounded answer.

- **`tests/graph/nodes/test_generate_answer_or_rag.py`**

  - **Purpose**: Verifies the initial routing logic.
    - **`'Not rewritten'` path**: Confirms that for a new user query, a `SystemMessage` is injected to guide the LLM, tools are bound, and the `has_been_rewritten` flag is correctly maintained.
    - **`'Already rewritten'` path**: Ensures that if the query has been rewritten, the `SystemMessage` is skipped, preserving the conversation's integrity.

- **`tests/graph/nodes/test_rewrite_question.py`**

  - **Purpose**: Checks that the query rewrite prompt is correctly formed from the original user question and that the output state properly includes the new `AIMessage` and sets `has_been_rewritten=True`.

- **`tests/graph/routing/test_document_grader.py`**

  - **Purpose**: Tests the conditional logic that decides whether retrieval is necessary.
    - **`'Bypass when already rewritten'`**: Ensures the grader is skipped if the question was already improved.
    - **`'Route on relevant'`**: Routes to answer generation if the context is deemed relevant.
    - **`'Route on irrelevant'`**: Routes to query rewriting if the context is irrelevant.

- **`tests/graph/orchestration/test_graph.py`**

  - **Purpose**: Provides end-to-end integration tests for the whole graph, using a `MemorySaver` checkpointer and mocked dependencies.
    - **`'Direct answer path'`**: Verifies that a simple question flows through the graph and results in a direct `AIMessage` without tool calls.
    - **`'RAG path'`**: Confirms that a question requiring retrieval correctly triggers the tool call → context extraction → final answer flow.

- **`tests/api/test_app.py`**
  - **Purpose**:
    - **`'Root endpoint'`**: Verifies the `GET /` health check works as expected.
    - **`'Chat streaming contract'`**: Ensures the `POST /chat` endpoint streams Server-Sent Events with the correct `start`, `data`, and `end` events, validating the API contract.
