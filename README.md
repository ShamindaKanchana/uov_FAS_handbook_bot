# University of Vavuniya – Faculty of Applied Science Handbook Bot

A Retrieval-Augmented Generation (RAG) assistant that answers academic questions about programmes, regulations and courses offered by the Faculty of Applied Science (FAS), University of Vavuniya. Instead of combing through the handbook PDF, students can query the bot in plain English and instantly receive relevant excerpts.

---

## ✨  Key Features

| Stage | Details |
| ----- | ------- |
| **Pre-processing** | PDF → structured JSON (sections, subsections, pages) via `pdf_parser.py`. |
| **Chunking** | `HandbookChunker` splits long sections into ~350-word overlapping chunks, enriching each with hierarchy & page metadata. |
| **Embedding** | `all-MiniLM-L6-v2` (Sentence-Transformers, 384-d) with cosine normalisation. |
| **Vector Store** | [Qdrant](https://qdrant.tech/) – local, file-based, persisted under `database/qdrant/`. |
| **Retrieval** | `QueryEngine` improves query (spelling fixes, term expansion), embeds it and performs semantic search; results optionally re-ranked. |
| **CLI Bot** | `python query_handbook.py` interactive shell delivering nicely formatted answers.

---

## 🗂️  Repository Layout

```
├── data/
│   ├── raw/              # Original PDF
│   ├── processed/        # Parsed JSON
│   └── chunks/           # `handbook_chunks.jsonl`
├── database/qdrant/      # Persistent Qdrant data
├── src/
│   ├── preprocessing/
│   │   ├── pdf_parser.py
│   │   └── chunker.py
│   ├── embedding/
│   │   ├── embedder.py
│   │   ├── config.py
│   │   └── qdrant_singleton.py
│   └── retrieval/
│       ├── retriever.py
│       └── reranker.py  # (placeholder)
└── query_handbook.py     # CLI entry-point
```

---

## ⚙️  End-to-End Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Handbook PDF   ├────►│  PDF Parser     ├────►│  JSON Chunks    │
│                 │     │  (pdf_parser)   │     │  (chunker)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Query     ├────►│  Query Engine   │◄────┤  Qdrant DB     │
│  (CLI/API)      │     │  (retriever)    │     │  (Vectors +     │
└─────────────────┘     └────────┬────────┘     │   Metadata)     │
                                 │              └─────────────────┘
                                 ▼
                         ┌─────────────────┐
                         │                 │
                         │  CLI Bot / LLM  │
                         │  (Response)     │
                         └─────────────────┘
```

1. **Parsing** – `pdf_parser.py` extracts headings & paragraphs.
2. **Chunking** – `chunker.py` cleans text, splits into meaningful chunks, adds metadata.
3. **Embedding** – `embedder.py` encodes each chunk and stores **vector + payload** in Qdrant.  The collection name is `uov_fas_handbook` (see `src/embedding/config.py`).
4. **Retrieval** – On each user question, `QueryEngine` embeds the improved query, fetches top-k similar chunks and returns them to the bot.
5. *(Future)* **Re-ranking/LLM generation** – `reranker.py` is reserved for cross-encoder or GPT-based answer synthesis.

---

## 🏗️  Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `sentence-transformers`, `qdrant-client`, `pdfplumber`, `tqdm`.

---

## 🚀  Usage Guide

### 1 ▪ Create chunks
```bash
python -m src.preprocessing.chunker \
  --input data/raw/fas_handbook.pdf \
  --output-dir data/chunks \
  --format jsonl
```

### 2 ▪ Embed & ingest (persistent)
```python
from src.embedding.embedder import TextEmbedder, load_document_chunks
from src.embedding.config import QDRANT_STORAGE_PATH

docs = load_document_chunks('data/chunks/handbook_chunks.jsonl')
embedder = TextEmbedder(storage_path=QDRANT_STORAGE_PATH)
embedder.add_documents(docs)
```

### 3 ▪ Chat
```bash
python query_handbook.py
```

or programmatically:
```python
from src.retrieval.retriever import QueryEngine
engine = QueryEngine()
print(engine.search('Subjects in IT degree', top_k=3))
```

---

## 🔧  Configuration
Edit `src/embedding/config.py` to tweak:
* `DEFAULT_MODEL` (embedding model)
* `DEFAULT_COLLECTION` (Qdrant collection name)
* `QDRANT_STORAGE_PATH` (database directory)
* Vector size, distance metric, batch size, etc.

---

## 📜  License
MIT – see `LICENSE`.

---

Built with ♥ to help FAS students navigate their curriculum effortlessly.
