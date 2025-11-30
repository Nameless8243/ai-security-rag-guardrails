import json
from pathlib import Path

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

from drift import record_retrieval, detect_drift
from mutation_detector import analyze_chunks_for_mutation
from context_guard import context_guard


DB_DIR = Path("chroma_db")


# -----------------------------------------------------------
# TRUST-BASED RERANKING
# -----------------------------------------------------------
def trust_rerank(docs):
    """
    Reorders documents so that:
    1. HIGH trust always comes first
    2. LOW trust comes afterwards
    3. Original similarity order inside each group is preserved
    """
    high = [d for d in docs if d.metadata.get("trust_level") == "high"]
    low  = [d for d in docs if d.metadata.get("trust_level") != "high"]

    return high + low


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------
def main():
    if not DB_DIR.exists():
        print("❌ No ChromaDB found. First run: python ingest.py")
        return

    embeddings = OllamaEmbeddings(model="mistral:7b")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    # Increased k → ensures policy chunks always appear
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    question = "What are the rules for sharing passwords when using AI systems?"

    print(f"❓ QUESTION:\n{question}\n")

    # --- Retrieve documents ---
    docs = retriever.invoke(question)

    # --- Trust-aware reranking ---
    docs = trust_rerank(docs)

    print("📥 RETRIEVED CONTEXT:\n")
    for d in docs:
        meta = d.metadata
        prev = d.page_content[:200].replace("\n", " ")
        print(f"[x] {meta}  → {prev}...\n")

    # --- Drift monitoring ---
    stats = record_retrieval([d.metadata["source"] for d in docs])
    drift_alerts = detect_drift(stats, [d.metadata["source"] for d in docs])

    if drift_alerts:
        for a in drift_alerts:
            print(f"🚨 DRIFT ALERT:\n   {a}\n")
    else:
        print("🟢 No retriever drift detected.\n")

    # --- Context guardrail (blocklist + embedding drift) ---
    guard = context_guard(docs)
    if guard:
        print(guard)
        print("⚠️ WARNING: Context looks suspicious, but continuing in WARN mode.\n")

    # --- Mutation detector (AI-based content check) ---
    mutation = analyze_chunks_for_mutation(docs)
    if mutation:
        print("🧬 MUTATION DETECTOR WARNING:")
        print("   " + mutation + "\n")
        print("⚠️ We DO NOT stop model execution — WARN mode.\n")

    # --- LLM answer generation ---
    llm = OllamaLLM(model="mistral:7b")
    combined = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an internal AI Security assistant.

Answer the following question briefly and with a strong security focus,
using ONLY the policy context provided.

[CONTEXT]
{combined}

[QUESTION]
{question}

[ANSWER]
"""

    print("🧠 Generating LLM answer...\n")
    answer = llm.invoke(prompt)
    print("💬 MODEL ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
