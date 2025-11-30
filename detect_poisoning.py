from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from utils import cosine_sim  # currently only used for consistency if needed later


DB_DIR = Path("chroma_db")


def load_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral:7b")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    return vectorstore


def detect_embedding_outliers(vectors, threshold: float = 2.5):
    """
    Simple Z-score based outlier detector on embedding norms.

    threshold: number of standard deviations above which a vector
               is considered suspicious.
    """
    norms = np.linalg.norm(vectors, axis=1)
    mean = norms.mean()
    std = norms.std()

    z_scores = (norms - mean) / (std + 1e-8)
    outliers = np.where(np.abs(z_scores) > threshold)[0]

    return outliers, norms, z_scores


def main():
    if not DB_DIR.exists():
        print("❌ No ChromaDB found (run: python ingest.py).")
        return

    vectorstore = load_vectorstore()

    # Read the full collection from ChromaDB
    docs = vectorstore._collection.get(include=["metadatas", "documents", "embeddings"])

    texts = docs["documents"]
    metadata = docs["metadatas"]
    vectors = np.array(docs["embeddings"])

    print(f"📦 Number of documents: {len(texts)}")

    # --- Embedding outlier detection ---
    outliers, norms, z_scores = detect_embedding_outliers(vectors)

    print("\n📊 EMBEDDING ANALYSIS:")
    for i, (meta, n, z) in enumerate(zip(metadata, norms, z_scores)):
        print(f"  [{i}] {meta}  | norm={n:.2f}  | z-score={z:.2f}")

    if len(outliers) > 0:
        print("\n🔥 POISONING SUSPICION – OUTLIERS DETECTED:")
        for idx in outliers:
            print(f"  → Document #{idx} is suspicious: {metadata[idx]['source']}")
    else:
        print("\n🟢 No outliers – embedding space looks clean.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
