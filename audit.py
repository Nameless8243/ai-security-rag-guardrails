import json
from datetime import datetime
from pathlib import Path


LOG_FILE = Path("audit_log.jsonl")


def audit_event(event_type: str, source: str | None = None,
                doc_hash: str | None = None, status: str | None = None) -> None:
    """
    Append a single ingest-related event in JSONL format.

    Used for:
    - tracking which documents were ingested
    - detecting duplicates
    - later forensic analysis of poisoning attempts
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        "source": source,
        "doc_hash": doc_hash,
        "status": status,
    }

    # Append mode: each event is a separate JSON line
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"📝 Audit log: {entry}")
