from pathlib import Path
from langchain_ollama import OllamaEmbeddings
import json

BASELINE_FILE = Path("baseline_embedding.json")


def build_baseline() -> None:
    """
    Build a baseline embedding from the official AI Security Policy.

    This baseline is later used by the context guard to detect
    embedding drift (sudden semantic changes in the policy space).
    """
    text = Path("data/good_policy.txt").read_text(encoding="utf-8")

    embeddings = OllamaEmbeddings(model="mistral:7b")
    vec = embeddings.embed_query(text)

    BASELINE_FILE.write_text(
        json.dumps({"embedding": vec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Baseline embedding created:", BASELINE_FILE)


if __name__ == "__main__":
    build_baseline()
