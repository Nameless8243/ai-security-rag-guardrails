from pathlib import Path
import hashlib

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from audit import audit_event

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")


def compute_hash(text: str) -> str:
    """Compute a stable hash of the document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_trust_level(path: Path) -> str:
    """
    Infer a rough trust level from the filename.

    - poisoned_policy.txt → "low"
    - everything else     → "high" (for this lab)
    """
    name = path.name.lower()
    if "poisoned" in name:
        return "low"
    return "high"


def load_docs():
    """Load all .txt documents from the data/ directory with metadata."""
    docs = []
    for path in DATA_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        doc_hash = compute_hash(text)
        trust_level = infer_trust_level(path)

        docs.append(
            {
                "text": text,
                "hash": doc_hash,
                "metadata": {
                    "source": str(path.name),
                    "trust_level": trust_level,
                    "doc_hash": doc_hash,
                },
            }
        )
    return docs


def main():
    docs = load_docs()

    embeddings = OllamaEmbeddings(model="mistral:7b")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    # Read existing metadata from Chroma to support deduplication
    existing_docs = vectorstore._collection.get(include=["metadatas"])
    existing_hashes = {
        meta.get("doc_hash")
        for meta in existing_docs["metadatas"]
        if meta and meta.get("doc_hash")
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    new_texts: list[str] = []
    new_meta: list[dict] = []

    for d in docs:
        if d["hash"] in existing_hashes:
            audit_event(
                event_type="duplicate",
                source=d["metadata"]["source"],
                doc_hash=d["hash"],
                status="skipped",
            )
            continue

        audit_event(
            event_type="ingest",
            source=d["metadata"]["source"],
            doc_hash=d["hash"],
            status="added",
        )

        chunks = splitter.split_text(d["text"])
        for chunk in chunks:
            new_texts.append(chunk)
            new_meta.append(d["metadata"])

    if new_texts:
        vectorstore.add_texts(texts=new_texts, metadatas=new_meta)
        print(f"✅ Ingest complete. New chunks: {len(new_texts)}")
    else:
        print("ℹ️ No new documents ingested.")


if __name__ == "__main__":
    main()
