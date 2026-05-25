import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


# BGE QUERY PREFIX

BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# EMBEDDING MODEL

class LocalEmbedder:

    def __init__(self, model_name=EMBEDDING_MODEL):

        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self):

        return self.model.get_embedding_dimension()

    # DOCUMENT EMBEDDINGS

    def encode_documents(
        self,
        texts,
        batch_size=32
    ):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype("float32")

    # QUERY EMBEDDINGS

    def encode_query(self, query):

        prefixed_query = BGE_QUERY_PREFIX + query

        embedding = self.model.encode(
            [prefixed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype("float32")[0]