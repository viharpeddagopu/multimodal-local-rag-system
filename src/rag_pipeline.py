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

    # CHECK IF QUERY NEEDS VISUAL GROUNDING

    def needs_visual_grounding(
        self,
        query
    ):

        visual_keywords = [
            "figure",
            "diagram",
            "image",
            "architecture",
            "chart",
            "graph",
            "table",
            "visual",
            "shown",
            "illustration"
        ]

        query_lower = query.lower()

        return any(
            keyword in query_lower
            for keyword in visual_keywords
        )

    # ASK QUESTION

    def ask(
        self,
        query
    ):

        # RETRIEVE RELEVANT CHUNKS

        retrieval_results = self.retriever.retrieve(
            query
        )

        retrieved_chunks = [
            result["chunk"]
            for result in retrieval_results
        ]

        # VISUAL GROUNDING

        visual_context = ""

        needs_visual = (
            self.needs_visual_grounding(query)
        )

        if (
            self.vlm
            and retrieved_chunks
            and needs_visual
        ):

            print(
                "\n=== RUNNING VLM VISUAL GROUNDING ==="
            )

            top_chunk = retrieved_chunks[0]

            try:

                # IMPORTANT:
                # CALLING INSTANCE METHOD CORRECTLY

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

                print("\n=== VISUAL CONTEXT ===")
                print(visual_context)

            except Exception as e:

                print(
                    f"VLM visual grounding failed: {e}"
                )

        else:

            print(
                "\n=== SKIPPING VLM "
                "(TEXT-ONLY QUERY) ==="
            )

        # FINAL QUERY

        final_query = query + visual_context

        print("\n=== FINAL QUERY TO QWEN ===")
        print(final_query)

        # GENERATE FINAL ANSWER

        answer = self.generator.generate(
            query=final_query,
            retrieved_chunks=retrieved_chunks
        )

        # SOURCES

        sources = sorted(
            set(
                f"{chunk.document} - Page {chunk.page}"
                for chunk in retrieved_chunks
            )
        )

        # FINAL RESPONSE

        result = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_results": retrieval_results,
            "visual_context": visual_context,
            "used_vlm": needs_visual
        }

        return result