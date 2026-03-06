"""
ingestion/embedder.py — Steps 3 + 4: embed all chunks and store in FAISS.

Every chunk — text, table caption, image caption — is embedded the same way
using text_content. That's the payoff of the captioning step: visuals become
searchable text without any special-case handling here.

We use text-embedding-3-small over CLIP because CLIP is capped at ~77 tokens
(image-native design), which truncates financial captions with tables, numbers,
and multi-sentence descriptions. text-embedding-3-small handles full captions.

FAISS IndexFlatIP with L2-normalised vectors gives cosine similarity — more
stable than L2 distance for comparing semantic embeddings of different lengths.

The index and chunk list are pickled separately so ingestion runs once and
query time just calls load().
"""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY, EMBED_MODEL, EMBED_DIM, VECTOR_STORE_DIR
from utils.schemas import DocumentChunk

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_FILE  = Path(VECTOR_STORE_DIR) / "index.faiss"
CHUNKS_FILE = Path(VECTOR_STORE_DIR) / "chunks.pkl"


def embed_and_store(chunks: list[DocumentChunk]) -> faiss.Index:
    """Embed all chunks and save the FAISS index + chunk list to disk."""
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}")

    texts   = [c.text_content for c in chunks]
    vectors = embed_batch(texts)

    arr = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(arr)   # normalise so inner product equals cosine similarity

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(arr)

    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index ({index.ntotal} vectors) → {VECTOR_STORE_DIR}")
    return index


def load() -> tuple[faiss.Index, list[DocumentChunk]]:
    """Load a previously built index and chunk list from disk."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError("No index found. Run: python main.py ingest")
    index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded index: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Call the OpenAI embeddings API in batches of 100 to stay within API limits."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[i : i + batch_size],
        )
        all_vecs.extend([item.embedding for item in resp.data])
    return all_vecs
