# University of Vavuniya – Faculty of Applied Science Handbook Bot

A Retrieval-Augmented Generation (RAG) assistant that answers academic questions about programmes, regulations, and courses offered by the Faculty of Applied Science (FAS), University of Vavuniya. This AI-powered assistant provides instant, accurate responses by combining semantic search with large language models.

![Demo](https://img.shields.io/badge/Demo-Available-success) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Current Status (Last Updated: July 2025)

### ✅ Implemented Features
- **Document Processing**: PDF parsing and text extraction
- **Vector Database**: Local Qdrant instance for efficient semantic search
- **API Integration**: Cohere for response generation
- **Web Interface**: Simple Flask-based web UI

### ⚠️ Known Limitations
- **Content Coverage**: 
  - ✅ **Bio Science Department**: Most content has been processed and is available
  - 🚧 **Physical Science Department**:  Not yet processed

- **Response Quality**: 
  - Responses may vary in quality based on the available context
  - Some specific queries might return incomplete or generic responses

### 🔄 Update Process
- Existing issues will identify and fix with the time 

## ✨  Key Features

### 🔍 Advanced Retrieval
- **Semantic Search**: Find relevant information using state-of-the-art embeddings
- **Context-Aware**: Understands academic context and course structures
- **Source Attribution**: Every response includes references to the original handbook content

### 🤖 Intelligent Generation
- **Natural Responses**: Human-like answers using Cohere's advanced language models
- **Contextual Understanding**: Maintains conversation context for follow-up questions
- **Precision-Tuned**: Optimized for academic and regulatory content

### 🛠️ Technical Highlights
- **Modular Architecture**: Clean separation of retrieval and generation components
- **Efficient Processing**: Optimized chunking and embedding pipeline
- **Local First**: Runs entirely on your machine with optional cloud components
- **Extensible**: Easy to update with new handbook versions or additional data sources

---

## 🗂️  Repository Structure

```
uov_fas_handbook_bot/
├── data/
│   ├── raw/                  # Original PDF documents
│   ├── processed/            # Parsed and structured JSON
│   └── chunks/               # Pre-processed text chunks
├── database/qdrant/          # Vector database storage
├── src/
│   ├── preprocessing/        # Document processing
│   │   ├── pdf_parser.py     # PDF to structured data
│   │   └── chunker.py        # Text segmentation
│   │
│   ├── embedding/            # Text embedding components
│   │   ├── embedder.py       # Document embedding logic
│   │   └── qdrant_singleton.py
│   │
│   ├── retrieval/            # Search and retrieval
│   │   ├── retriever.py      # Query processing
│   │   └── reranker.py       # Result re-ranking
│   │
│   └── generation/           # Response generation
│       ├── generator.py      # Response generation logic
│       └── nlp.py            # NLP utilities
│
├── query_handbook.py         # CLI interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚙️  How It Works

### End-to-End Pipeline

```mermaid
flowchart TD
    A[Handbook PDF] -->|Parse| B[PDF Parser]
    B -->|Extract Text| C[Text Chunks]
    C -->|Generate Embeddings| D[Qdrant Vector DB]
    E[User Query] -->|Process| F[Query Engine]
    F -->|Search| D
    D -->|Retrieve Context| F
    F -->|Generate Response| G[Response Generation]
    G -->|Natural Language| H[User]
```

### Key Components

1. **Document Processing**
   - Extracts and structures content from PDF handbooks
   - Splits content into manageable chunks with metadata
   - Generates semantic embeddings for efficient retrieval

2. **Query Processing**
   - Handles natural language queries
   - Performs semantic search against the vector database
   - Ranks and retrieves most relevant content chunks

3. **Response Generation**
   - Formats retrieved context into coherent responses
   - Uses Cohere's language model for natural-sounding answers
   - Includes source attribution for verification

### Data Flow

```mermaid
sequenceDiagram
    participant User
    participant App as Web/CLI App
    participant QueryEngine
    participant Qdrant
    participant LLM as Cohere LLM

    User->>App: Asks question
    App->>QueryEngine: Process query
    QueryEngine->>Qdrant: Search similar vectors
    Qdrant-->>QueryEngine: Return top-k chunks
    QueryEngine->>LLM: Generate response with context
    LLM-->>QueryEngine: Generated response
    QueryEngine-->>App: Formatted response
    App-->>User: Display response with sources
```

### Implementation Details

1. **Parsing** – `pdf_parser.py` extracts headings & paragraphs
2. **Chunking** – `chunker.py` processes text into meaningful chunks with metadata
3. **Embedding** – `embedder.py` creates vector representations using `all-MiniLM-L6-v2`
4. **Storage** – Vectors and metadata stored in Qdrant collection `uov_fas_handbook`
5. **Retrieval** – `QueryEngine` handles semantic search and response generation
6. **Generation** – Uses Cohere's API for natural language responses
---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Cohere API key (for response generation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/uov-fas-handbook-bot.git
   cd uov-fas-handbook-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Cohere API key
   ```

### Usage

#### Command Line Interface
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
