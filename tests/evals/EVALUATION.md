# RAG Evaluation Guide

This guide explains how to use the evaluation suite to measure and improve the performance of your RAG system. It is designed for both quick local checks and comprehensive cloud-based experiments.

## Evaluation Workflow

Follow these steps to evaluate your RAG application.

### Step 1: Generate a Baseline Dataset

First, you need a dataset of questions and answers derived from your source documents. This script creates it for you.

- **What it does**: Reads all `.txt` files in `tests/data`, generates question/answer pairs using an LLM, and saves them to a CSV file.
- **When to run it**: Run this once to create your initial `synthetic_dataset.csv`. Re-run it whenever you significantly change the documents in `tests/data`.

```bash
# Ensure your OPENAI_API_KEY is set in .env
source .venv/bin/activate

# Optional: Specify the model for generation (defaults to gpt-4o-mini)
export DEEPEVAL_SYNTH_MODEL=gpt-4o-mini

python scripts/evals/generate_goldens.py
```

- **Output**: `tests/evals/data/synthetic_dataset.csv`

---

### Step 2: Run Local Tests (Fast Feedback)

These local tests run quickly and are perfect for getting fast feedback during development without relying on external services like LangSmith.

#### A) Mock Graph Test (Quick & Isolated)

- **What it does**: Tests the RAG metrics (Faithfulness, Relevancy, etc.) against a _mock_ graph. This test does not run your actual RAG agent but uses a simplified, predictable stand-in.
- **When to run it**: Use this for CI or as a quick sanity check to ensure the evaluation logic and metrics are working correctly.

```bash
source .venv/bin/activate
pytest tests/evals/test_rag_evaluation.py -v
```

#### B) Real Graph Test (End-to-End Local Check)

- **What it does**: Tests the full RAG agent end-to-end, but replaces the Pinecone vector store with a simple, local `BM25Retriever`.
- **When to run it**: This is the most important local test. Run it after making changes to your graph logic to see how it impacts performance before deploying or running expensive cloud evaluations.

```bash
source .venv/bin/activate

# Run on the first 3 test cases by default
pytest tests/evals/test_rag_real_graph.py -v

# Run on the first 10 test cases
export DEEPEVAL_TEST_LIMIT=10
pytest tests/evals/test_rag_real_graph.py -v
```

---

### Step 3 (Optional): Augment Dataset for Robustness Testing

To understand how your RAG system handles different user phrasing, you can create an "augmented" version of your dataset.

- **What it does**: Takes the baseline dataset and creates three versions ("slices") of each question: the original (`canonical`), a more conversational version (`humanized`), and a slightly harder version (`challenging`).
- **When to run it**: Run this when you want to perform more rigorous testing to see how robust your system is to variations in user input. This is often done before a major release or after significant changes to the retrieval or question-rewriting logic.

```bash
source .venv/bin/activate
python scripts/evals/augment_slices.py
```

- **Output**: `tests/evals/data/synthetic_dataset_slices.csv`

---

### Step 4 (Optional): Run Full Experiment in LangSmith (Cloud Evaluation)

For the most comprehensive and shareable results, you can run the evaluation against the LangSmith platform.

- **What it does**: Uploads the augmented dataset to LangSmith (you must do this manually first), then runs your full RAG agent against each test case and logs the results, traces, and metric scores to your LangSmith project.
- **When to run it**: Use this to benchmark major versions of your agent, compare different prompts or models, and share detailed results with your team.

```bash
source .venv/bin/activate

# Set up your LangSmith credentials
export LANGSMITH_API_KEY=...
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_PROJECT="Your-Project-Name"
export OPENAI_API_KEY=...

# Run the experiment on the first 10 examples from your LangSmith dataset
export EVAL_LIMIT=10
python scripts/evals/run_langsmith_experiment.py
```

## Reference: Metrics Evaluated

All tests measure the core "RAG Triad" plus contextual relevance:

1.  **Faithfulness**: Does the answer stick to the facts from the retrieved context? (Prevents hallucination)
2.  **Answer Relevancy**: Is the answer relevant to the user's question?
3.  **Contextual Recall**: Did the retriever find all the necessary information from the ground-truth context?
4.  **Contextual Relevancy**: Is the retrieved information relevant to the question? (Prevents retrieving useless documents)

## Reference: Configuration

- **Test Case Limit (Local)**: Use `DEEPEVAL_TEST_LIMIT` to control how many rows from the CSV are used in `pytest`.
- **Test Case Limit (LangSmith)**: Use `EVAL_LIMIT` to control how many examples from the LangSmith dataset are used.
- **Evaluation Model**: Use `DEEPEVAL_SYNTH_MODEL` or `DEEPEVAL_METRIC_MODEL` to set which LLM (`gpt-4o-mini`, `gpt-4-turbo`, etc.) is used for generating data and calculating metrics.
