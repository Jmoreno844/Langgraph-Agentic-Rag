import os
from pathlib import Path
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.models import GPTModel
import csv

# Load environment variables from .env file
load_dotenv()


def generate():
    """
    Generates a synthetic dataset for RAG evaluation from documents in tests/data.
    """
    print("🚀 Starting synthetic dataset generation...")

    # Configure model via env var; default to a cost-effective and capable model
    model_name = os.getenv("DEEPEVAL_SYNTH_MODEL", "gpt-5-nano")
    print(f"🧠 Using model: {model_name}")

    # Some OpenAI models only support the default temperature (1). Use GPTModel to control it.
    deepeval_model = GPTModel(model=model_name, temperature=1)
    synthesizer = Synthesizer(model=deepeval_model)

    # Point to the directory with your text files
    data_dir = Path("./tests/data")
    doc_paths = list(data_dir.glob("*.txt"))

    if not doc_paths:
        print("❌ No .txt files found in tests/data. Aborting.")
        return

    doc_paths_str = [str(p) for p in doc_paths]
    print(f"📚 Found {len(doc_paths_str)} documents to process.")

    print(f"📝 Generating up to 2 goldens per context...")

    # Generate synthetic goldens directly from document paths
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=doc_paths_str,
        include_expected_output=True,
        max_goldens_per_context=2,
    )
    print(f"✅ Generated {len(goldens)} initial goldens.")

    # Convert Golden objects to dictionaries and rename keys for the new schema.
    # We can't add attributes to the Pydantic 'Golden' model directly.
    processed_goldens = []
    for g in goldens:
        # .model_dump() is the Pydantic v2 way to get a dict
        golden_dict = g.model_dump()
        processed_goldens.append(
            {
                "input": golden_dict.get("input"),
                "expected_generation_output": golden_dict.get("expected_output"),
                "expected_retrieval_context": golden_dict.get("context"),
                "source_file": golden_dict.get(
                    "source_file"
                ),  # Ensure this is carried over
            }
        )

    # Manually write to CSV to control column names
    output_path = "./evals"
    output_file = "synthetic_dataset.csv"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, output_file)

    if not processed_goldens:
        print("⚠️ No goldens generated, skipping CSV write.")
        return

    # Use the keys from the first dictionary as headers
    fieldnames = processed_goldens[0].keys()
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_goldens)

    print(f"\n✅ Synthetic dataset generated successfully!")
    print(f"📂 Saved to {full_path}")
    print(f"📈 Dataset contains {len(processed_goldens)} goldens.")


if __name__ == "__main__":
    generate()
