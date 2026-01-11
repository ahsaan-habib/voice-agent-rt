"""Retrieval from rag-grounded (hybrid + rerank), minus generation — the voice
loop streams its own answer."""
from __future__ import annotations

from rag_grounded.ingest.chunker import Chunk
from rag_grounded.retrieval.hybrid import HybridRetriever
from rag_grounded.retrieval.rerank import Reranker


class Retriever:
    def __init__(self, top_k: int = 4):
        self.hybrid = HybridRetriever()
        self.reranker = Reranker()
        self.top_k = top_k

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        candidates = self.hybrid.vectors.get(self.hybrid.retrieve(query))
        return self.reranker.rank(query, candidates)[: self.top_k]
