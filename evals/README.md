# RAG Evaluation Setup

This directory contains the evaluation setup for your LangGraph RAG system using DeepEval.

## Files Overview

- `synthetic_dataset.csv` - Generated evaluation dataset with 32 test cases
- `../tests/evals/test_rag_evaluation.py` - Comprehensive pytest evaluation suite
- `../scripts/test_rag_quick.py` - Quick verification script

## Dataset Structure

The synthetic dataset contains the following columns:
- `input` - The question to ask the RAG system
- `expected_output` - The ideal answer (ground truth)
- `context` - The ground truth context that should be retrieved
- `actual_output` - Filled by tests (what your system actually answered)
- `retrieval_context` - Filled by tests (documents your system retrieved)
- `source_file` - Which document the test case came from

## Running Evaluations

### Quick Verification (Recommended First)
```bash
source .venv/bin/activate
python scripts/test_rag_quick.py
```

This runs a single test case to verify everything works before running the full suite.

### Full Test Suite
```bash
source .venv/bin/activate
pytest tests/evals/test_rag_evaluation.py -v
```

### Run Specific Metrics
```bash
# Run only faithfulness tests
source .venv/bin/activate
pytest tests/evals/test_rag_evaluation.py::test_rag_evaluation -k "faithfulness" -v

# Run with different thresholds
source .venv/bin/activate
pytest tests/evals/test_rag_evaluation.py -v --tb=short
```

## Metrics Evaluated

The evaluation measures the **RAG Triad** plus additional metrics:

1. **Faithfulness** - Does the answer stick to facts from retrieved context?
2. **Answer Relevancy** - Is the answer relevant to the original question?
3. **Contextual Recall** - Did the system retrieve all relevant documents?
4. **Contextual Relevancy** - Are the retrieved documents actually relevant?

## Configuration

- **Minimum Score Threshold**: 0.7 (configurable in the test file)
- **Model**: gpt-4o-mini (configurable via DEEPEVAL_SYNTH_MODEL env var)
- **Test Cases**: 32 automatically generated from your documents

## Troubleshooting

### Common Issues

1. **"Dataset not found"**
   ```bash
   source .venv/bin/activate
   export DEEPEVAL_SYNTH_MODEL=gpt-4o-mini
   python scripts/generate_eval_dataset.py
   ```

2. **Database connection issues**
   - Ensure your `.env` file has correct AWS_DB_URL
   - Make sure the database is running

3. **OpenAI API issues**
   - Check your OPENAI_API_KEY in `.env`
   - Ensure you have sufficient API credits

4. **Model not supported errors**
   - Use `gpt-4o-mini` or `gpt-4` for evaluation
   - Set via: `export DEEPEVAL_SYNTH_MODEL=gpt-4o-mini`

## Interpreting Results

- **Scores ≥ 0.7**: Generally acceptable performance
- **Faithfulness < 0.7**: Your system may be hallucinating
- **Answer Relevancy < 0.7**: Answers not addressing the question properly
- **Contextual Recall < 0.7**: Missing relevant documents
- **Contextual Relevancy < 0.7**: Retrieving irrelevant documents

## Next Steps

1. Run the quick test to verify everything works
2. Run the full test suite to get baseline scores
3. Use results to identify areas for improvement
4. Re-run evaluations after making changes to measure improvement
