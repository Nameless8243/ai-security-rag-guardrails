# 🛡️ AI Security Lab – RAG Poisoning, Drift Detection & Guardrails

This project is an educational AI Security lab demonstrating how to build a **secure Retrieval-Augmented Generation (RAG)** pipeline with:

- Document hashing & ingestion audit logs  
- Outlier detection for poisoned embeddings  
- Retriever drift monitoring  
- Context guardrails  
- LLM-based mutation detection  
- Strict policy enforcement from a single “official” document  

It is intentionally simplified to help beginners understand real-world AI security concepts.

---

## 📁 Project Structure

```
AI_SECURITY_LAB/
│
├── data/
│   ├── ai_security_overview.txt      # official policy docs
│   ├── data_protection.txt
│   ├── encryption_policy.txt
│   ├── iam_rules.txt
│   ├── incident_response.txt
│   ├── good_policy.txt               # authoritative security policy
│   └── poisoned_policy.txt           # deliberately malicious policy
│
├── audit.py                          # append-only JSONL logging
├── baseline_embedding.py             # baseline vector for drift detection
├── context_guard.py                  # context filters + embedding drift check
├── detect_poisoning.py               # embedding outlier detector
├── drift.py                          # retriever drift monitor
├── ingest.py                         # policy ingestion + hashing + split
├── mutation_detector.py              # LLM-based mutation checking (WARN mode)
├── rag_query.py                      # main RAG + guardrail pipeline
├── utils.py                          # cosine similarity helper
└── requirements.txt
```

---

## ✔️ Prerequisites

- Python **3.11+** recommended  
- [**Ollama**](https://ollama.com/) installed and running  
- A local model pulled, for example:

```bash
ollama pull mistral:7b
```

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## ▶️ Running the Lab

### **1. Build the baseline embedding** (from the official policy)

```bash
python baseline_embedding.py
```

---

### **2. Ingest all policy documents into Chroma**

```bash
python ingest.py
```

The script will:

- compute SHA-256 hash for each document  
- write entries into `audit_log.jsonl`  
- split, embed, and store documents in `chroma_db/`  


---

### **3. Run the secure RAG pipeline with all guardrails**

```bash
python rag_query.py
```

You will see:

- Retrieved context (including the malicious policy)  
- Retriever drift monitoring  
- Context guardrail warnings  
- Mutation detector warnings  
- Final secured LLM answer (preferring the strict policy)  

---

## 🔍 Security Modules Overview

### **✔ Document Hashing & Audit Log**
- SHA-256 hash for every document
- Append-only JSONL audit trail

### **✔ Embedding Outlier Detection**
Detects poisoned or anomalous embeddings using Z-score thresholds.

### **✔ Retriever Drift Detection**
Flags:
- dominant documents (e.g., 95%+ retrieval rate)
- sudden new documents

### **✔ Context Guardrails**
Blocks:
- jailbreak-style instructions  
- policy-breaking language  
- embedding drift vs baseline

### **✔ LLM Mutation Detection**
Uses a local LLM to detect:
- rewritten policy content  
- harmful “exceptions”
- permissions that contradict the official policy

---

## ⚠️ Disclaimer

This project is **for educational purposes only**.  
It intentionally simplifies core AI security concepts:

- RAG poisoning  
- Embedding integrity checks  
- Guardrails  
- Drift monitoring  
- LLM mutation analysis  

**Do not reuse this code in production systems.**

---

## 📄 License

MIT License  
Copyright © 2025  

