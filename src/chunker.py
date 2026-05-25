from dataclasses import dataclass
from pathlib import Path

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# CHUNK DATA STRUCTURE

@dataclass
class Chunk:

    document: str

    page: int

    text: str

    image_path: Path


# SIMPLE RECURSIVE CHUNKING

def split_text(
    text,
    chunk_size,
    chunk_overlap
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


# CREATE CHUNKS

def create_chunks(pages):

    chunks = []

    for page in pages:

        split_texts = split_text(
            text=page.text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        for split_text_piece in split_texts:

            chunk = Chunk(
                document=page.document,
                page=page.page,
                text=split_text_piece,
                image_path=page.image_path
            )

            chunks.append(chunk)

    return chunks