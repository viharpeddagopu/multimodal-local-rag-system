import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

import streamlit as st

from src.config import (
    PAPERS_DIR,
    FAISS_INDEX_DIR
)

from src.embedder import LocalEmbedder
from src.vector_store import FaissStore
from src.retriever import Retriever
from src.generator import QwenGenerator
from src.rag_pipeline import RAGPipeline
from src.vlm import MoondreamVLM


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Multimodal Local RAG",
    layout="wide"
)

st.title("Multimodal Local RAG System")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Indexed Papers")

pdf_files = list(PAPERS_DIR.glob("*.pdf"))

for pdf in pdf_files:

    st.sidebar.markdown(f"- {pdf.name}")


# =========================================================
# LOAD PIPELINE
# =========================================================

@st.cache_resource
def load_pipeline():

    # ---------------------------------------------
    # Embedding model
    # ---------------------------------------------

    embedder = LocalEmbedder()

    # ---------------------------------------------
    # Load FAISS index
    # ---------------------------------------------

    vector_store = FaissStore.load(
        FAISS_INDEX_DIR
    )

    # ---------------------------------------------
    # Retriever
    # ---------------------------------------------

    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        use_reranker=True
    )

    # ---------------------------------------------
    # Generator
    # ---------------------------------------------

    generator = QwenGenerator()

    # ---------------------------------------------
    # Vision Language Model
    # ---------------------------------------------

    vlm = None # device constraints

    # ---------------------------------------------
    # RAG Pipeline
    # ---------------------------------------------

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        vlm=vlm
    )

    return rag_pipeline


# =========================================================
# INITIALIZE PIPELINE
# =========================================================

with st.spinner("Loading models and vector database..."):

    pipeline = load_pipeline()


# =========================================================
# QUESTION INPUT
# =========================================================

query = st.text_input(
    "Ask a question about the papers:"
)


# =========================================================
# GENERATE RESPONSE
# =========================================================

if query:

    with st.spinner("Generating answer..."):

        result = pipeline.ask(query)

    # =====================================================
    # ANSWER
    # =====================================================

    st.subheader("Answer")

    st.write(result["answer"])

    # =====================================================
    # VISUAL GROUNDING
    # =====================================================

    if result["visual_context"]:

        st.subheader("Visual Grounding")

        st.info(result["visual_context"])

    # =====================================================
    # SOURCES
    # =====================================================

    st.subheader("Sources")

    for source in result["sources"]:

        st.write(f"- {source}")

    # =====================================================
    # RETRIEVED CHUNKS
    # =====================================================

    st.subheader("Retrieved Chunks")

    retrieval_results = result["retrieval_results"]

    for retrieval in retrieval_results:

        chunk = retrieval["chunk"]

        with st.expander(
            f"{chunk.document} - Page {chunk.page}"
        ):

            # -----------------------------------------
            # Scores
            # -----------------------------------------

            st.caption(
                f"Dense Score: "
                f"{retrieval['dense_score']:.4f}"
            )

            if retrieval["rerank_score"] is not None:

                st.caption(
                    f"Rerank Score: "
                    f"{retrieval['rerank_score']:.4f}"
                )

            # -----------------------------------------
            # Chunk Text
            # -----------------------------------------

            st.write(chunk.text)

            # -----------------------------------------
            # Page Image
            # -----------------------------------------

            if chunk.image_path.exists():

                st.image(
                    str(chunk.image_path),
                    caption=(
                        f"{chunk.document} "
                        f"- Page {chunk.page}"
                    ),
                    use_container_width=True
                )