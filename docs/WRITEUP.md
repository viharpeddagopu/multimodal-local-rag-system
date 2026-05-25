# Multimodal Local RAG System — Technical Writeup

**Author:** Vihar Peddagopu  
**Email:** viharpeddagopu@gmail.com

---

# 1. Project Overview

The goal of this project was to build a fully local Retrieval-Augmented Generation (RAG) system capable of answering questions over machine learning research papers using semantic retrieval and local language models.

The system processes PDF papers, extracts and chunks text, generates embeddings locally, stores them inside a FAISS vector database, retrieves relevant chunks during querying, reranks them for better relevance, and generates grounded answers using a local language model.

The architecture also supports optional multimodal grounding through a Vision Language Model (VLM), where rendered PDF page images can be linked to retrieved chunks for visual context understanding.

The project was designed with a strong focus on:
- modularity
- local inference
- explainability
- retrieval quality
- maintainable architecture

---

# 2. Papers Used

The system was tested using publicly available machine learning papers:

- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

These papers were selected because they:
- belong to similar technical domains
- contain semantically rich concepts
- are suitable for retrieval-based evaluation

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
Reranker
    ↓
Local LLM / Optional VLM
    ↓
Generated Answer
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
- Qwen for text generation
- Moondream2 for optional visual grounding

Instead of using only a VLM for the entire pipeline, a hybrid architecture was chosen.

### Why Qwen for Text Generation

Qwen was selected because:
- it provides stronger text reasoning performance
- generates faster responses for text-only queries
- consumes fewer resources compared to running a VLM continuously
- works well for retrieval-grounded generation

Most user queries over research papers are primarily text-based, so using a dedicated text LLM was more efficient and practical.

### Why Moondream for Visual Grounding

Moondream2 was selected because:
- it is relatively lightweight compared to larger VLMs
- supports image-based reasoning
- can process rendered PDF page images
- enables future support for diagrams, equations, and figures

### Why Not Use Only a VLM

Using only a VLM for every query would:
- increase inference latency significantly
- consume much more memory
- create high CPU/GPU usage even for simple text questions

Because of this, a hybrid design was chosen:
- Qwen handles normal text generation
- Moondream can optionally provide visual grounding when needed

This produced a better balance between:
- performance
- resource usage
- multimodal capability

---

## 3.4 FAISS for Retrieval

FAISS was used as the vector database because:
- it is lightweight
- optimized for similarity search
- efficient for local semantic retrieval
- easy to integrate with embedding pipelines

Embeddings are stored locally so the index does not need to be rebuilt on every application startup.

---

## 3.5 Retrieval + Reranking

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

## 3.6 Multimodal Support

The architecture supports optional multimodal reasoning through Moondream2.

PDF pages are rendered into images during ingestion and linked to retrieved chunks.

This enables future visual grounding using:
- diagrams
- equations
- figures
- page layouts

However, live VLM inference was disabled by default because of high latency on CPU-only systems.

This was an intentional engineering tradeoff to maintain responsiveness.

---

# 4. What Worked Well

Several parts of the system worked well during testing:

- semantic retrieval quality was strong
- reranking noticeably improved relevance
- modular architecture simplified debugging
- local inference removed dependency on external APIs
- retrieved chunk visualization improved explainability
- FAISS indexing reduced repeated computation overhead

The retriever consistently identified the correct paper and relevant pages for technical queries.

---

# 5. Problems Faced During Development

## 5.1 Hardware Limitations

One major challenge was running local VLMs on limited hardware.

System configuration:

| Component | Specification |
|---|---|
| RAM | 16 GB |
| GPU | NVIDIA RTX 2050 |
| VRAM | 4 GB |

Because of limited VRAM and memory:
- larger VLMs became difficult to run
- model loading was slow
- inference latency increased significantly
- RAM usage became very high during multimodal inference

Some larger models could not run efficiently on the available hardware.

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

As a result, multimodal inference was kept optional instead of running on every query.

---

## 5.4 Serialization and Metadata Issues

While storing chunk metadata alongside the FAISS index, issues occurred involving:
- JSON serialization
- WindowsPath objects
- metadata persistence

This required restructuring how metadata was saved and loaded.

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
- dependency debugging
- hardware-aware AI engineering

One major takeaway was understanding how important modularity and clean architecture are when building larger AI systems.

Another important learning was realizing how expensive multimodal inference can become on consumer hardware, and how engineering tradeoffs are necessary to balance performance and usability.

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
```

For each query:
1. the retriever searched the FAISS database
2. the top retrieved chunk was analyzed
3. the predicted source document was compared against the expected document

The evaluation focused primarily on:
- retrieval correctness
- semantic relevance
- grounded answer generation

---

# 8. Evaluation Results

The retriever achieved:

```text
Accuracy: 100%
Correct Predictions: 6/6
```

The evaluation demonstrated that:
- semantic retrieval was functioning correctly
- the retriever consistently identified the correct source paper
- reranking improved relevance of retrieved chunks
- the generated answers remained grounded in retrieved context

More difficult semantic queries such as:
- “How does the transformer remove recurrence?”
- “What is bidirectional pretraining?”

were also retrieved correctly, indicating that the system was performing semantic retrieval rather than simple keyword matching.

---

# 9. Conclusion

This project successfully implemented a fully local multimodal-ready RAG pipeline using:
- semantic retrieval
- reranking
- FAISS vector search
- local language models
- optional visual grounding

The project provided practical experience in:
- AI system design
- retrieval engineering
- multimodal inference
- debugging local AI pipelines
- optimizing systems under hardware constraints

The final system achieved strong retrieval quality while maintaining a modular and explainable architecture.