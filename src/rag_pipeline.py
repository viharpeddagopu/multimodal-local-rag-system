from src.generator import QwenGenerator
from src.retriever import Retriever
from src.vlm import MoondreamVLM


# RAG PIPELINE

class RAGPipeline:

    def __init__(
        self,
        retriever,
        generator,
        vlm=None
    ):

        self.retriever = retriever

        self.generator = generator

        self.vlm = vlm

    # ASK QUESTION

    def ask(
        self,
        query
    ):

        # Retrieve relevant chunks

        retrieval_results = self.retriever.retrieve(
            query
        )

        retrieved_chunks = [
            result["chunk"]
            for result in retrieval_results
        ]

        # Visual grounding (optional)

        visual_context = ""

        if self.vlm and retrieved_chunks:

            top_chunk = retrieved_chunks[0]

            try:

                visual_answer = (
                    self.vlm.answer_visual_question(
                        image_path=top_chunk.image_path,
                        question=query
                    )
                )

                visual_context = (
                    f"\n\nVisual Context:\n"
                    f"{visual_answer}"
                )

            except Exception as e:

                print(
                    f"VLM visual grounding failed: {e}"
                )

        # Generate final answer

        answer = self.generator.generate(
            query=query + visual_context,
            retrieved_chunks=retrieved_chunks
        )

        # Sources

        sources = sorted(
            set(
                f"{chunk.document} - Page {chunk.page}"
                for chunk in retrieved_chunks
            )
        )

        # Final response

        result = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_results": retrieval_results,
            "visual_context": visual_context
        }

        return result
    

# =========================================================
# HYBRID QWEN + MOONDREAM ARCHITECTURE
# =========================================================
#
# This system uses a collaborative hybrid architecture where
# both Qwen and Moondream2 work together instead of replacing
# one another.
#
# Workflow:
#
# User Query
# ↓
# Retriever fetches relevant text chunks
# ↓
# Moondream2 receives:
# - rendered PDF page image
# - user query
# ↓
# Moondream2 extracts visual understanding from the page
# (figures, diagrams, equations, layouts, visual structure)
# ↓
# The visual understanding is converted into textual context
# ↓
# The visual context is appended to the original query
# ↓
# Qwen receives:
# - retrieved chunks
# - original query
# - visual context extracted by Moondream2
# ↓
# Qwen generates the final grounded answer