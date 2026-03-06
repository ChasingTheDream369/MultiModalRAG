"""
retrieval/retriever.py — Step 5: hybrid retrieval with FAISS + BM25 fused via RRF.

Why RRF over weighted score combination:
  Weighted combination (e.g. 0.7 * semantic + 0.3 * bm25) requires min-max
  normalising raw scores first. If one chunk has an outlier-high BM25 score,
  normalisation compresses all other BM25 scores toward zero, killing the keyword
  signal for the rest of the results.

  Reciprocal Rank Fusion works on ranks, not raw scores:
      RRF(chunk) = 1/(k + rank_semantic) + 1/(k + rank_bm25)
  Ranks are always 1..N regardless of score magnitude, so neither retriever
  can dominate. k=60 is the standard constant (Cormack et al. 2009).
"""
import logging
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

from config import OPENAI_API_KEY, EMBED_MODEL, TOP_K
from utils.schemas import DocumentChunk

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

RRF_K = 60   # standard constant — dampens the impact of very high ranks


class HybridRetriever:
    """
    FAISS semantic search + BM25 keyword search fused with Reciprocal Rank Fusion.

    BM25 is built at init from all chunk text_content — this means captions from
    the VLM are indexed for keyword search too, not just semantic search.
    """

    def __init__(self, index: faiss.Index, chunks: list[DocumentChunk]):
        self.index  = index
        self.chunks = chunks
        tokenized   = [c.text_content.lower().split() for c in chunks]
        self.bm25   = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[DocumentChunk]:
        """
        Return top-K chunks by RRF score.

        Candidate pool is 4x top_k for both retrievers — casting a wide net
        before fusion means we don't miss chunks that rank well in only one retriever.
        """
        pool = min(top_k * 4, len(self.chunks))

        # FAISS semantic ranking
        q_vec = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        _, faiss_idxs = self.index.search(q_vec, pool)
        semantic_ranking = [i for i in faiss_idxs[0] if i != -1]

        # BM25 keyword ranking
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        bm25_ranking = np.argsort(bm25_scores)[::-1][:pool].tolist()

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(semantic_ranking):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_idxs = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [self.chunks[i] for i in top_idxs]


def embed_query(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding
