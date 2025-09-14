import os
import csv
import hashlib
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Using OpenAI client directly for robust text rewrites
try:
    from openai import OpenAI  # Newer OpenAI SDK
except ImportError:
    OpenAI = None


load_dotenv()


def _stable_family_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _ensure_openai_client():
    if OpenAI is None:
        raise RuntimeError(
            "openai package not installed. Please `pip install openai` in your environment."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY or export it in your shell."
        )
    return OpenAI()


def _chat_complete(client, model: str, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1,
    )
    return (resp.choices[0].message.content or "").strip()


def _humanize_question(client, model: str, question: str) -> str:
    system = (
        "You rewrite questions into natural, conversational user phrasing. "
        "Preserve the exact intent and facts. Do not add or remove constraints."
    )
    user = (
        "Rewrite the following question so it sounds like a typical human user.\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Keep meaning identical and answerable from the same context.\n"
        "- Prefer simple, friendly phrasing.\n"
        "- Output ONLY the rewritten question."
    )
    return _chat_complete(client, model, system, user)


def _challenging_question(client, model: str, question: str) -> str:
    system = (
        "You rewrite questions to be slightly more challenging while keeping the same factual target answer. "
        "Require a small inference or implied detail, but do not change what the correct answer is."
    )
    user = (
        "Rewrite the following question to require mild reasoning or an implied detail,"
        " while preserving the same final answer.\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Do NOT change the factual target or constraints.\n"
        "- Keep it concise and human.\n"
        "- Output ONLY the rewritten question."
    )
    return _chat_complete(client, model, system, user)


def _read_rows(src_csv: Path) -> List[Dict[str, str]]:
    with src_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def _write_rows(dst_csv: Path, rows: List[Dict[str, str]], fieldnames: List[str]):
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def augment_dataset(
    src_path: str = "./tests/evals/data/synthetic_dataset.csv",
    dst_path: str = "./tests/evals/data/synthetic_dataset_slices.csv",
    model_name: str | None = None,
    limit: int | None = None,
):
    client = _ensure_openai_client()
    model = model_name or os.getenv(
        "AUGMENT_MODEL", os.getenv("DEEPEVAL_SYNTH_MODEL", "gpt-5-nano")
    )

    src_csv = Path(src_path)
    if not src_csv.exists():
        raise FileNotFoundError(f"Source CSV not found at {src_csv}")

    base_rows = _read_rows(src_csv)
    if limit is not None:
        base_rows = base_rows[: max(0, int(limit))]

    out_rows: List[Dict[str, str]] = []

    # Prepare fieldnames (union of existing plus our new columns)
    existing_fields = set(base_rows[0].keys()) if base_rows else set()
    extra_fields = {"slice", "family_id"}
    fieldnames = list(existing_fields.union(extra_fields))

    for idx, row in enumerate(base_rows, start=1):
        question = (row.get("input") or "").strip()
        if not question:
            continue

        family_id = _stable_family_id(question)

        # Canonical
        canonical_row = dict(row)
        canonical_row["slice"] = "canonical"
        canonical_row["family_id"] = family_id
        out_rows.append(canonical_row)

        # Humanized
        try:
            human_q = _humanize_question(client, model, question)
        except Exception as e:
            human_q = question  # fallback to original
        human_row = dict(row)
        human_row["input"] = human_q
        human_row["slice"] = "humanized"
        human_row["family_id"] = family_id
        # Clear run-dependent fields if present
        if "actual_output" in human_row:
            human_row["actual_output"] = ""
        if "retrieval_context" in human_row:
            human_row["retrieval_context"] = ""
        out_rows.append(human_row)

        # Challenging
        try:
            chall_q = _challenging_question(client, model, question)
        except Exception:
            chall_q = question
        chall_row = dict(row)
        chall_row["input"] = chall_q
        chall_row["slice"] = "challenging"
        chall_row["family_id"] = family_id
        if "actual_output" in chall_row:
            chall_row["actual_output"] = ""
        if "retrieval_context" in chall_row:
            chall_row["retrieval_context"] = ""
        out_rows.append(chall_row)

        if idx % 5 == 0:
            print(f"Processed {idx} base rows -> {len(out_rows)} total rows so far...")

    _write_rows(Path(dst_path), out_rows, fieldnames)
    print(f"\n✅ Wrote augmented dataset with slices to: {dst_path}")
    print(f"📈 Base rows: {len(base_rows)}  ->  Output rows: {len(out_rows)}")


if __name__ == "__main__":
    # Simple CLI via env vars; modify as needed
    src = os.getenv("AUGMENT_SRC", "./tests/evals/data/synthetic_dataset.csv")
    dst = os.getenv("AUGMENT_DST", "./tests/evals/data/synthetic_dataset_slices.csv")
    model = os.getenv("AUGMENT_MODEL", None)
    lim = os.getenv("AUGMENT_LIMIT", None)
    augment_dataset(
        src_path=src, dst_path=dst, model_name=model, limit=int(lim) if lim else None
    )
