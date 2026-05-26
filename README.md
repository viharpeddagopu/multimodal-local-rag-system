# Multimodal Local RAG System

A fully local multimodal Retrieval-Augmented Generation (RAG) system for querying research papers using semantic retrieval, reranking, adaptive multimodal routing, and visual grounding.

The system processes PDF research papers, builds a FAISS vector database over extracted text chunks, retrieves relevant contexts using dense embeddings, reranks them using a cross-encoder, and generates grounded answers using local language models.

For visually-oriented queries such as figures, diagrams, charts, and tables, the system selectively invokes a Vision Language Model (VLM) for visual grounding before generating the final response.

All inference runs locally without paid APIs.

---

# Features

- Fully local inference pipeline
- PDF text extraction using PyMuPDF
- Recursive chunking with overlap
- Dense semantic retrieval using FAISS
- Cross-encoder reranking
- Local Qwen-based answer generation
- Moondream2 visual grounding support
- Adaptive multimodal routing
- Rendered PDF page grounding
- Streamlit-based interactive UI
- Explainable retrieval with retrieved chunks and source pages
- Docker support for reproducible environments
- No external API dependency

---

# System Architecture

```text
PDF Papers
    ↓
Document Processing
    ↓
Chunking
    ↓
Embeddings Generation
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Cross-Encoder Reranker
    ↓
Adaptive Routing
    ├── Text Query → Qwen
    └── Visual Query
            ↓
      Moondream2 Visual Grounding
            ↓
            Qwen
    ↓
Answer Generation
```

---

# Multimodal Workflow

In this architecture, Moondream2 is not responsible for generating the final answer. Instead, it acts as a visual context extraction module that augments the textual retrieval pipeline with image understanding.

The workflow is:

1. Relevant chunks are retrieved from the FAISS index
2. Associated PDF page images are preserved during retrieval
3. For visual queries, Moondream2 analyzes the rendered page image
4. Visual context is extracted from figures and diagrams
5. The visual grounding is appended to the retrieved textual context
6. Qwen generates the final grounded response

This hybrid architecture performs significantly better than relying entirely on a small VLM for reasoning and long-context answer generation.

---

# Why Hybrid Architecture?

Moondream2 is optimized primarily for:

- image understanding
- visual grounding
- figure interpretation

Qwen is optimized for:

- textual reasoning
- long-context understanding
- grounded answer generation

Using them collaboratively allows:

- efficient multimodal reasoning
- better answer quality
- reduced unnecessary VLM inference
- improved system responsiveness

---

# Adaptive Multimodal Routing

The system uses adaptive multimodal routing where visual grounding is selectively invoked only for visually-oriented queries such as:

- figures
- diagrams
- charts
- tables
- visual architectures

Text-only queries bypass the VLM entirely and use only:

- retrieval
- reranking
- Qwen generation

This significantly reduces unnecessary multimodal inference latency.

---

# Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vector Database | FAISS |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Text LLM | Qwen2.5-1.5B-Instruct |
| VLM | Moondream2 |
| PDF Processing | PyMuPDF |
| Backend | Python |

---

# Project Structure

```text
multimodal-local-rag-system/
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│   ├── text-query-output.png
│   ├── retrieved_chunks-text-query.png
│   ├── visual-query-output.png
│   └── retrieved-chunks-visual-query.png
│
├── data/
│   ├── papers/
│   └── faiss_index/
│
├── src/
│   ├── __init__.py
│   ├── chunker.py
│   ├── config.py
│   ├── document_processor.py
│   ├── embedder.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── vlm.py
│
├── build_index.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How the System Works

## 1. Document Processing

PDF papers are processed using PyMuPDF:

- text is extracted page-by-page
- each page is rendered into an image
- metadata is preserved for retrieval grounding

---

## 2. Chunking

Extracted text is split into overlapping chunks to:

- preserve semantic continuity
- improve retrieval quality
- reduce context fragmentation

---

## 3. Embedding Generation

Each chunk is converted into dense vector embeddings using:

```text
BAAI/bge-small-en-v1.5
```

These embeddings represent semantic meaning numerically.

---

## 4. FAISS Vector Index

Embeddings are stored locally inside a FAISS vector database for:

- efficient similarity search
- scalable retrieval
- low-latency querying

---

## 5. Retrieval and Reranking

The pipeline performs:

1. Dense retrieval using cosine similarity
2. Cross-encoder reranking for relevance refinement

This improves retrieval accuracy significantly compared to single-stage retrieval.

---

## 6. Visual Grounding

For visual queries:

- associated PDF page images are retrieved
- Moondream2 analyzes figures and diagrams
- extracted visual understanding is appended to the textual context

---

## 7. Final Answer Generation

Qwen generates the final grounded response using:

- retrieved textual chunks
- reranked context
- optional visual grounding

Retrieved source chunks and corresponding pages are displayed in the UI for explainability.

---

# How to Read the Code

| File | Purpose |
|---|---|
| `config.py` | Central configuration and paths |
| `document_processor.py` | PDF extraction and page rendering |
| `chunker.py` | Text chunking logic |
| `embedder.py` | Embedding generation |
| `vector_store.py` | FAISS persistence and retrieval |
| `retriever.py` | Dense retrieval + reranking |
| `generator.py` | Qwen response generation |
| `rag_pipeline.py` | Adaptive multimodal orchestration |
| `vlm.py` | Moondream2 visual grounding |
| `streamlit_app.py` | Streamlit user interface |

---

# Quick Start (Recommended)

The easiest way to run the project is using Docker.

## 1. Clone Repository

```bash
git clone https://github.com/viharpeddagopu/multimodal-local-rag-system
cd multimodal-local-rag-system
```

---

## 2. Build Docker Image

```bash
docker build -t multimodal-rag .
```

---

## 3. Run Docker Container

```bash
docker run -p 8501:8501 multimodal-rag
```

Open:

```text
http://localhost:8501
```

Models are automatically downloaded from HuggingFace during first execution and are not packaged into the Docker image.

Initial startup may take several minutes due to first-time model downloads and local model initialization.

---

# Manual Local Setup

## 1. Create Virtual Environment

It is strongly recommended to use a clean virtual environment to avoid dependency conflicts with existing global or Conda-installed packages.

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Build FAISS Index

```bash
python build_index.py
```

---

## 4. Run Application

```bash
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

The file watcher is disabled to avoid compatibility issues between Streamlit and PyTorch/transformer-based model loading on certain systems.

---

# Add Research Papers

Two research papers are already included for testing.

To add additional PDFs, place them inside:

```text
data/papers/
```

---

# Screenshots

## Text Query Output

![Text Query Output](assets/text-query-output.png)

---

## Retrieved Chunks - Text Query

![Retrieved Chunks Text Query](assets/retrieved_chunks-text-query.png)

---

## Visual Query Output (VLM Grounding)

![Visual Query Output](assets/visual-query-output.png)

---

## Retrieved Chunks - Visual Query

![Retrieved Chunks Visual Query](assets/retrieved-chunks-visual-query.png)

---

# Hardware Notes

- The project is designed primarily for local CPU-based inference
- Large local models can significantly increase RAM usage
- Visual grounding is computationally expensive on CPU-only systems
- Visual queries are substantially slower than text-only queries

---

# Troubleshooting

## FAISS Index Not Found

Run:

```bash
python build_index.py
```

before launching the application.

---

## Slow First Startup

The first launch downloads and initializes:

- embedding models
- rerankers
- Qwen
- Moondream2

Subsequent runs are significantly faster due to local caching.

---

## High CPU or RAM Usage

Large local models may consume substantial system memory during inference.

Visual queries involving diagrams and figures may require significantly longer inference times on consumer-grade hardware due to local VLM execution.

---

# Contact

**Vihar Peddagopu**  
Email: viharpeddagopu@gmail.com
