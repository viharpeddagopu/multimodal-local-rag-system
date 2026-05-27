# Multimodal Local RAG System — Technical Writeup

**Author:** Vihar Peddagopu  
**Email:** viharpeddagopu@gmail.com

---

# 1. Project Overview

The goal of this project was to build a fully local Retrieval-Augmented Generation (RAG) system capable of answering questions over machine learning research papers using semantic retrieval and local language models.

The system processes PDF papers, extracts and chunks text, generates embeddings locally, stores them inside a FAISS vector database, retrieves relevant chunks during querying, reranks them for better relevance, and generates grounded answers using a local language model.

The architecture also supports multimodal grounding through a Vision Language Model (VLM), where rendered PDF page images can be linked to retrieved chunks for visual context understanding.

The project was designed with a strong focus on:

- modularity
- local inference
- explainability
- retrieval quality
- maintainable architecture
- multimodal extensibility

---

# 2. Papers Used

The system was tested using publicly available machine learning papers:

- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

---

# 3. Architecture and Design Choices

The system follows a modular retrieval-first architecture:

```text
PDF Papers
    ↓
Document Processing
    ↓
Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Database
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

## 3.1 Modular Structure

One major design choice was building the project using separate modules instead of a single large script.

Different responsibilities were separated into independent files:

- PDF processing
- chunking
- embeddings
- retrieval
- reranking
- generation
- vector storage
- visual grounding

This improved:

- readability
- debugging
- maintainability
- scalability

One important thing learned during this project was how useful proper project structure and modularity are in larger AI systems.

---

## 3.2 Local Inference

The entire system runs locally without paid APIs.

Reasoning:

- lower operational cost
- better privacy
- full control over models
- understanding local AI deployment challenges

Smaller local models were intentionally selected to balance:

- inference quality
- memory usage
- hardware limitations

---

## 3.3 Choice of Qwen and Moondream

The system uses:

- Qwen2.5-1.5B-Instruct for text reasoning and answer generation
- Moondream2 for visual grounding

Initially, I explored using a single VLM-based architecture for the entire pipeline. However, during experimentation I observed that smaller VLMs are optimized primarily for visual understanding rather than strong long-context textual reasoning.

A VLM with around 2B parameters dedicates a significant portion of its capacity toward:

- image encoding
- visual feature alignment
- multimodal attention
- image-text fusion

As a result, using a small VLM alone for both:

- image understanding
- final long-form textual reasoning

produced weaker performance compared to combining:

- a dedicated text LLM
- with a specialized VLM

Because of this, I implemented a hybrid multimodal architecture:

- Moondream2 performs visual grounding
- Qwen performs the final grounded answer generation

The pipeline works as follows:

1. Relevant chunks are retrieved from FAISS
2. Associated rendered PDF page images are preserved
3. For visual queries, Moondream analyzes the page image
4. Visual understanding is extracted from figures and diagrams
5. The extracted visual context is appended to the retrieved textual context
6. Qwen generates the final grounded response

This architecture produced significantly better:

- textual reasoning
- retrieval grounding
- response quality
- runtime efficiency

compared to using only a small VLM for the entire pipeline.

---

## 3.4 Adaptive Hybrid Multimodal Routing

After implementing the hybrid architecture, another important issue became apparent during testing.

Running the VLM for every query caused:
- unnecessary multimodal inference
- high CPU/GPU utilization
- increased memory consumption
- much slower response times

However, many user queries were purely textual and did not require image understanding.

Because of this, I redesigned the pipeline into an adaptive hybrid multimodal architecture.

The system now selectively invokes the VLM only for visually-oriented queries containing keywords related to concepts such as:
- figures
- diagrams
- charts
- tables
- visual architectures

Text-only queries bypass the VLM entirely and use:
- retrieval
- reranking
- Qwen generation

This significantly improved:
- responsiveness
- runtime efficiency
- resource utilization

while still preserving multimodal capability for visual questions.

This became one of the most important architectural improvements in the project because it balanced:
- multimodal capability
- practical usability on consumer hardware

The current implementation uses keyword-based routing for simplicity and efficiency.

A better future implementation would involve semantic query understanding, where the system predicts whether a query requires visual grounding instead of relying only on explicit keywords.

---

## 3.5 FAISS for Retrieval

FAISS was used as the vector database because:

- it is lightweight
- optimized for similarity search
- efficient for local semantic retrieval
- easy to integrate with embedding pipelines

Embeddings are stored locally so the index does not need to be rebuilt on every application startup.

---

## 3.6 Preprocessing and Ingestion Design

Another important design choice was separating:

- document ingestion
- from query-time inference

During preprocessing:

- PDFs are extracted
- pages are rendered into images
- chunks are generated
- embeddings are created
- FAISS indexes are built
- metadata is stored locally

This preprocessing happens only once during index construction.

At query time, the system only:

- loads the already-built index
- retrieves embeddings
- performs reranking
- generates answers

This significantly improved runtime efficiency because expensive preprocessing operations do not need to repeat for every query.

Separating ingestion from querying made the system:

- faster
- cleaner architecturally
- more scalable
- easier to debug

---

## 3.7 Retrieval + Reranking

The retrieval pipeline was designed in two stages:

### Dense Retrieval

FAISS retrieves semantically similar chunks using embeddings.

### Cross-Encoder Reranking

A reranker model reorders retrieved chunks based on relevance to the query.

This improved answer grounding significantly because:

- dense retrieval provides fast semantic recall
- reranking improves precision

The reranked chunks consistently produced better final answers compared to retrieval alone.

---

# 4. What Worked Well

Several parts of the system worked well during testing:

- semantic retrieval quality was strong
- reranking noticeably improved relevance
- modular architecture simplified debugging
- local inference removed dependency on external APIs
- adaptive multimodal routing reduced unnecessary VLM inference
- retrieved chunk visualization improved explainability
- FAISS indexing reduced repeated computation overhead
- hybrid LLM + VLM design improved answer quality

The retriever consistently identified the correct paper and relevant pages for technical queries.

The adaptive hybrid architecture also improved practical usability significantly compared to always running the VLM.

---

# 5. Problems Faced During Development

## 5.1 Hardware Limitations and Local Multimodal Inference

One major challenge was running local multimodal models on consumer hardware.

System configuration:

| Component | Specification |
|---|---|
| RAM | 16 GB |
| GPU | NVIDIA RTX 2050 |
| VRAM | 4 GB |

Initially, running only the Qwen text LLM worked relatively well.

However, after integrating Moondream2 visual grounding into the pipeline, the system became significantly slower during multimodal inference.

I observed:

- very high CPU usage
- large memory consumption
- long response times
- occasional inference stalls during VLM execution

This helped me understand an important practical limitation of fully local multimodal AI systems:

multimodal inference is substantially more computationally expensive than text-only inference.

The issue became more apparent on consumer-grade hardware with limited VRAM and memory capacity.

This challenge directly influenced several architectural decisions:

- hybrid LLM + VLM design
- adaptive multimodal routing
- optional VLM invocation
- preprocessing optimizations

These changes were implemented specifically to reduce unnecessary multimodal computation while preserving multimodal capability.

---

## 5.2 Moondream and Transformers Compatibility Issues

Several compatibility issues occurred between:

- transformers versions
- Moondream
- dependency packages

These caused:

- import errors
- model loading issues
- inference failures

This required debugging dependency versions and adjusting model-loading configurations.

---

## 5.3 Expensive VLM Inference

Visual-language inference was significantly more computationally expensive than text-only inference.

Passing images into the VLM caused:

- high CPU usage
- long response times
- increased memory consumption

As a result, multimodal inference was redesigned into an adaptive system instead of running on every query.

This became an important optimization for practical usability.

---

## 5.4 Serialization and Metadata Issues

While storing chunk metadata alongside the FAISS index, issues occurred involving:

- JSON serialization
- WindowsPath objects
- metadata persistence

This required restructuring how metadata was saved and loaded.

---

## 5.5 Environment and Dependency Issues

Another challenge involved managing Python environments and dependency compatibility.

During development, several issues occurred involving:

- virtual environments
- Conda conflicts
- PyTorch compatibility
- transformers versions
- PyMuPDF installation
- local package resolution

These environment inconsistencies occasionally caused:

- import failures
- dependency conflicts
- model-loading issues

To improve reproducibility and simplify setup, Docker support was added to the project.

Docker helped provide:

- isolated environments
- dependency consistency
- reproducible builds
- simpler deployment

This also reduced the likelihood of “works on my machine” issues during project execution on different systems.

---

# 6. What I Learned

This project helped me understand several important engineering concepts:

- modular AI system design
- proper project structure
- reusable and maintainable code organization
- retrieval pipeline engineering
- semantic search systems
- FAISS vector databases
- reranking architectures
- local inference optimization
- multimodal AI pipelines
- VLM integration
- adaptive multimodal routing
- dependency debugging
- Docker-based reproducibility
- hardware-aware AI engineering

One major takeaway was understanding how important modularity and clean architecture are when building larger AI systems.

Another important learning was realizing how expensive multimodal inference can become on consumer hardware, and how engineering tradeoffs are necessary to balance:

- performance
- usability
- latency
- hardware limitations

---

# 7. Evaluation Methodology

The system was evaluated using manually created technical queries:

```text
1. What is attention?
2. What is self-attention?
3. What is BERT?
4. What is masked language modeling?
5. How does the transformer remove recurrence?
6. What is bidirectional pretraining?
7. What does Figure 2 show?
```

For each query:

1. the retriever searched the FAISS database
2. retrieved chunks were inspected
3. reranked results were evaluated
4. generated responses were analyzed
5. source grounding was verified manually

For visual queries:

- rendered page images were inspected
- VLM routing activation was verified
- visual grounding quality was evaluated manually

The evaluation focused primarily on:

- retrieval correctness
- semantic relevance
- grounded answer generation
- multimodal routing correctness
- visual grounding integration

---

# 8. Evaluation Results

The retriever achieved strong semantic retrieval quality across all tested queries.

Observed results:

- correct paper retrieval for all test queries
- relevant chunk grounding
- improved retrieval precision after reranking
- successful adaptive VLM routing
- successful visual grounding integration
- grounded answer generation using retrieved context

More difficult semantic queries such as:

- “How does the transformer remove recurrence?”
- “What is bidirectional pretraining?”

were also retrieved correctly, indicating that the system was performing semantic retrieval rather than simple keyword matching.

Visual grounding queries involving figures and diagrams successfully activated the VLM pipeline and appended visual context into the final generation workflow.

The evaluation demonstrated that:

- semantic retrieval was functioning correctly
- reranking improved relevance significantly
- adaptive routing reduced unnecessary multimodal computation
- the generated answers remained grounded in retrieved context

---

# 9. Conclusion

This project successfully implemented a fully local adaptive hybrid multimodal RAG pipeline using:

- semantic retrieval
- reranking
- FAISS vector search
- local language models
- visual grounding
- adaptive multimodal routing

The final architecture combines:

- Qwen for textual reasoning
- Moondream2 for visual understanding

while selectively invoking multimodal inference only when required.

The project provided practical experience in:

- retrieval engineering
- multimodal AI systems
- local inference optimization
- system design tradeoffs
- dependency management
- Docker-based reproducibility
- hardware-aware AI engineering

One of the biggest lessons learned during this project was understanding the practical tradeoffs between:

- multimodal capability
- inference latency
- hardware constraints
- architectural efficiency

The final system achieved strong retrieval quality while maintaining:

- explainability
- modularity
- local execution
- scalable architecture
- multimodal extensibility
