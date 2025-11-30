import json
from pathlib import Path
from collections import defaultdict

STATS_FILE = Path("retriever_stats.json")


def load_stats():
    if not STATS_FILE.exists():
        return defaultdict(int)
    try:
        data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        return defaultdict(int, data)
    except Exception:
        return defaultdict(int)


def save_stats(stats) -> None:
    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def record_retrieval(doc_sources):
    """
    Update retrieval statistics.

    doc_sources example:
        ["good_policy.txt", "poisoned_policy.txt"]
    """
    stats = load_stats()

    for src in doc_sources:
        stats[src] += 1

    save_stats(stats)
    return stats


def detect_drift(stats, current_docs):
    """
    Detect retriever drift via:
    1) Dominance: if a single document accounts for >95% of all hits, flag it.
    2) New document: if a previously unseen document suddenly appears.
    """
    if not stats:
        return None

    total = sum(stats.values())
    alerts: list[str] = []

    # --- 1) Dominance check ---
    for doc, count in stats.items():
        ratio = count / total
        if ratio > 0.95:   # relaxed threshold suitable for small toy setups
            alerts.append(f"🔥 DRIFT SUSPECTED: '{doc}' is too dominant ({ratio*100:.1f}%).")

    # --- 2) New document check ---
    for doc in current_docs:
        if doc not in stats:
            alerts.append(f"⚠️ NEW DOCUMENT: '{doc}' has not appeared in retriever stats before.")

    return alerts if alerts else None
