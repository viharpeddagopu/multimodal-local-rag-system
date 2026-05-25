from sentence_transformers import CrossEncoder

from src.config import TOP_K
from src.embedder import LocalEmbedder
from src.vector_store import FaissStore


# RETRIEVER

class Retriever:

    def __init__(
        self,
        vector_store,
        embedder,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker=False
    ):

        self.vector_store = vector_store

        self.embedder = embedder

        self.use_reranker = use_reranker

        self.reranker = None

        if use_reranker:

            self.reranker = CrossEncoder(
                reranker_model
            )

    # RETRIEVE

    def retrieve(
        self,
        query,
        top_k=TOP_K
    ):

        # Query embedding

        query_embedding = self.embedder.encode_query(
            query
        )

        # Dense retrieval

        retrieved = self.vector_store.search(
            query_embedding,
            top_k=top_k * 3
        )

        # Optional reranking

        if self.use_reranker and self.reranker:

            pairs = [
                (query, chunk.text)
                for chunk, _ in retrieved
            ]

            rerank_scores = self.reranker.predict(
                pairs
            )

            reranked = []

            for (chunk, dense_score), rerank_score in zip(
                retrieved,
                rerank_scores
            ):

                reranked.append({
                    "chunk": chunk,
                    "dense_score": float(dense_score),
                    "rerank_score": float(rerank_score)
                })

            reranked.sort(
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            return reranked[:top_k]

        # Without reranking

        results = []

        for chunk, score in retrieved[:top_k]:

            results.append({
                "chunk": chunk,
                "dense_score": float(score),
                "rerank_score": None
            })

        return results