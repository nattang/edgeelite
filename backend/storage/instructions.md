# 📦 Storage & Embedding System — Implementation Guide

## 🔍 Project Context

This module is part of a Qualcomm Intern HaQathon project built under the theme **"AI for Everyone, Everywhere."** The goal is to build an **on-device personal AI assistant** that runs entirely offline using **Snapdragon X Elite**.

The assistant combines:

- **OCR pipeline**: captures screen frames and extracts visible text
- **Audio pipeline**: captures microphone input and transcribes speech
- **Storage + Embedding (this module)**: stores, processes, and embeds raw text for semantic retrieval
- **LLM + UI layer**: provides query-based assistance using recent context

### 🧠 System Overview

- The user starts a **session** → OCR and Audio pipelines run in parallel
- Each pipeline emits raw text asynchronously, tied to a shared `session_id`
- This module stores those events in SQLite, then embeds and indexes them
- At inference time, it enables **vector-based semantic search** over past screen/audio content
- The UI queries this module to assemble context for the LLM

---

## ✅ Components

| Component        | Description                                        |
| ---------------- | -------------------------------------------------- |
| **SQLite**       | Persists raw input (OCR/audio) and embedded chunks |
| **FAISS**        | In-memory vector store for fast semantic search    |
| **LangChain**    | Wraps FAISS + embedding model                      |
| **MiniLM-L6-v2** | Default open-source embedding model (384-dim)      |

---

## 🧱 Folder Structure

```
/storage
  ├── db.py              # SQLite schema + insert/fetch logic
  ├── interface.py       # Public API: store, process, search
  ├── cleaner.py         # Text cleaning and chunking logic
  ├── faiss_store.py     # FAISS + LangChain wrapper
  └── utils.py           # UUID, timestamp, JSON helpers
```

---

## 🗃️ SQLite Schema

### `raw_events`

Stores raw text blocks from OCR or Audio pipelines.

```sql
CREATE TABLE IF NOT EXISTS raw_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source TEXT NOT NULL,
    ts DATETIME NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT
);
```

### `nodes`

Stores cleaned + embedded chunks from each raw input.

```sql
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    chunk TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY(event_id) REFERENCES raw_events(id)
);
```

---

## 🔁 System Flow

1. **Frontend generates `session_id`**
2. **OCR/Audio pipelines send raw text** (async) with timestamps and session ID
3. **`store_raw_event()`** stores each input to SQLite
4. Once session ends, **`process_session(session_id)`**:
   - Fetches all raw events
   - Cleans + chunks text
   - Embeds chunks
   - Stores to SQLite + FAISS
5. **`search_similar(query)`** enables retrieval during inference

---

## ⚙️ Public API

### `store_raw_event(...)`

```python
store_raw_event(
    session_id: str,
    source: str,  # 'ocr' or 'audio'
    ts: float,
    text: str,
    metadata: dict
) -> str
```

Saves raw event to `raw_events`.

---

### `process_session(...)`

```python
process_session(session_id: str) -> List[str]
```

Processes a full session's raw input:

- Cleans + chunks text
- Embeds each chunk
- Saves to `nodes` + FAISS

---

### `search_similar(...)`

```python
search_similar(query: str, k: int = 5) -> List[Tuple[str, str]]
```

Returns top-k similar chunks from FAISS with:

- `chunk`: matched chunk text
- `event_id`: source event

---

## 🔍 Embedding: MiniLM-L6-v2 via LangChain

We're using the HuggingFace model:

```
sentence-transformers/all-MiniLM-L6-v2
```

To load:

```python
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

To create the FAISS vector store:

```python
from langchain.vectorstores import FAISS
vector_store = FAISS.from_documents(docs, embedding_model)
vector_store.save_local("faiss_index/")
```

And reload:

```python
vector_store = FAISS.load_local("faiss_index/", embedding_model)
```

---

## 🧪 Example Flow

```python
from storage.interface import store_raw_event, process_session, search_similar

event_id = store_raw_event("abc123", "ocr", ts=..., text="Raw text", metadata={})
process_session("abc123")
matches = search_similar("what was I doing earlier?", k=5)
```

---

## ✅ Dev Notes

- `session_id` groups all raw input from a user session
- `id` uniquely identifies each raw text block
- `embedding` is stored as a `BLOB` in `nodes`
- FAISS index lives in-memory but is saved to disk after updates

---

## 📌 Next Steps

1. Implement `db.py`: create tables and insert/fetch logic
2. Implement `faiss_store.py`: init FAISS, add + search
3. Hook them via `interface.py`

You're ready to build!
