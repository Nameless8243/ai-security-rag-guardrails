import json
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from utils import cosine_sim

BASELINE_FILE = Path("baseline_embedding.json")

# Blocklist patterns for indirect prompt injection / context override
CONTEXT_BLOCKLIST = [
    "ignore previous",
    "forget previous instructions",
    "system override",
    "developer mode",
    "jailbreak",
    "reset all safety rules",
]


def context_guard(chunks):
    """
    Context-level guardrail:
    1) Scan the combined context for dangerous override patterns.
    2) Compare average context embedding to baseline to detect drift.

    Returns:
        - None if everything looks safe
        - str (message) if something looks suspicious
    """
    # --- 1) Combine all chunk texts and scan for blocklisted patterns ---
    joined = " ".join([c.page_content for c in chunks]).lower()

    for pattern in CONTEXT_BLOCKLIST:
        if pattern in joined:
            return f"❌ Context blocked ❗ forbidden pattern detected: '{pattern}'"

    # --- 2) Load baseline embedding ---
    if not BASELINE_FILE.exists():
        return "❌ Missing baseline embedding. Run: python baseline_embedding.py"

    baseline_vec = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["embedding"]

    embeddings = OllamaEmbeddings(model="mistral:7b")

    # --- 3) Compute embeddings for the current chunks ---
    chunk_embeddings = [embeddings.embed_query(c.page_content) for c in chunks]

    # Average embedding across all returned chunks
    avg_chunk = [
        sum(col) / len(chunk_embeddings)
        for col in zip(*chunk_embeddings)
    ]

    # --- 4) Drift detection via cosine similarity ---
    sim = cosine_sim(baseline_vec, avg_chunk)

    # Very relaxed threshold, because this is a tiny, artificial RAG setup.
    if sim < -0.40:
        return f"❌ Context blocked ❗ embedding drift detected (sim={sim:.2f})"

    return None
