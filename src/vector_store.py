import json
from pathlib import Path

import faiss
import numpy as np

from src.chunker import Chunk


# FAISS VECTOR STORE

class FaissStore:

    INDEX_FILE = "index.faiss"

    METADATA_FILE = "metadata.json"

    # INIT

    def __init__(
        self,
        dimension,
        index_dir
    ):

        self.dimension = dimension

        self.index_dir = Path(index_dir)

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.chunks = []

    # ADD EMBEDDINGS

    def add(
        self,
        embeddings,
        chunks
    ):

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    # SEARCH

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index < len(self.chunks):

                results.append(
                    (
                        self.chunks[index],
                        float(distance)
                    )
                )

        return results

    # SAVE

    def save(self):

        # Save FAISS index

        faiss.write_index(
            self.index,
            str(
                self.index_dir / self.INDEX_FILE
            )
        )

        # Save metadata

        metadata = {

            "dimension": self.dimension,

            "chunks": [

                {
                    "document": chunk.document,
                    "page": chunk.page,
                    "text": chunk.text,
                    "image_path": str(
                        chunk.image_path
                    )
                }

                for chunk in self.chunks
            ]
        }

        with open(
            self.index_dir / self.METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2
            )

    # LOAD

    @classmethod
    def load(
        cls,
        index_dir
    ):

        index_dir = Path(index_dir)

        # Load metadata

        with open(
            index_dir / cls.METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        # Create store

        store = cls(
            dimension=metadata["dimension"],
            index_dir=index_dir
        )

        # Load FAISS index

        store.index = faiss.read_index(
            str(
                index_dir / cls.INDEX_FILE
            )
        )

        # Load chunks

        store.chunks = [

            Chunk(
                document=chunk_data["document"],
                page=chunk_data["page"],
                text=chunk_data["text"],
                image_path=Path(
                    chunk_data["image_path"]
                )
            )

            for chunk_data in metadata["chunks"]
        ]

        return store