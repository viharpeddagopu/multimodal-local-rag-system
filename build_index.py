from pathlib import Path

from src.config import (
    PAPERS_DIR,
    FAISS_INDEX_DIR
)

from src.document_processor import extract_pdf_text
from src.chunker import create_chunks
from src.embedder import LocalEmbedder
from src.vector_store import FaissStore



# =========================================================
# LOAD PDF FILES
# =========================================================

pdf_paths = list(
    Path(PAPERS_DIR).glob("*.pdf")
)

print(f"Found {len(pdf_paths)} PDFs")


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

pages = extract_pdf_text(pdf_paths)

print(f"Extracted {len(pages)} pages")


# =========================================================
# CHUNKING
# =========================================================

chunks = create_chunks(pages)

print(f"Created {len(chunks)} chunks")


# =========================================================
# EMBEDDINGS
# =========================================================

embedder = LocalEmbedder()

chunk_texts = [
    chunk.text
    for chunk in chunks
]

embeddings = embedder.encode_documents(
    chunk_texts
)

print("Embeddings created")


# =========================================================
# FAISS INDEX
# =========================================================

vector_store = FaissStore(
    dimension=embedder.dimension,
    index_dir=FAISS_INDEX_DIR
)

vector_store.add(
    embeddings,
    chunks
)

vector_store.save()

print("FAISS index saved")