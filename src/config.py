from pathlib import Path


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
PAGE_IMAGES_DIR = DATA_DIR / "page_images"


# MODELS

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# CHUNKING

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# RETRIEVAL

TOP_K = 3


# GENERATION

MAX_NEW_TOKENS = 200