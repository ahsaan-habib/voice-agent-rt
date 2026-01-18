"""Retrieval from rag-grounded (hybrid + rerank), minus generation — the voice
loop streams its own answer."""
from __future__ import annotations

import asyncio
import re

from rag_grounded.ingest.chunker import Chunk
from rag_grounded.retrieval.hybrid import HybridRetriever
from rag_grounded.retrieval.rerank import Reranker


class Retriever:
    def __init__(self, top_k: int = 4, candidates: int = 20):
        # 20 candidates instead of rag-grounded's 30: ~fewer cross-encoder pairs,
        # golden-set recall moved 0.89 -> 0.88 for it
        self.hybrid = HybridRetriever(candidates=candidates)
        self.reranker = Reranker()
        self.top_k = top_k

    def warm(self) -> None:
        """A cold cross-encoder adds ~600 ms to the first request after idle —
        the request a person is most likely judging. Load and run it now."""
        self.search("how do I install laravel")

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        candidates = self.hybrid.vectors.get(self.hybrid.retrieve(query))
        return self.reranker.rank(query, candidates)[: self.top_k]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


class Prefetch:
    """Start retrieval on a partial transcript instead of waiting for ASR to
    finalise. If the final transcript matches the partial we searched, the
    result is already there; if not, the search is thrown away. A wasted
    vector search costs milliseconds of CPU, which is why this speculation
    is worth it (and speculating the LLM isn't)."""

    def __init__(self, retriever: Retriever, min_words: int = 4):
        self.retriever = retriever
        self.min_words = min_words
        self.text: str | None = None
        self.task: asyncio.Task | None = None
        self.wasted = 0
        self.on_result = None   # (text, ranked) -> None, used to speculate generation

    def on_partial(self, text: str) -> None:
        if len(text.split()) < self.min_words or _norm(text) == _norm(self.text or ""):
            return
        if self.task and not self.task.done():
            self.task.cancel()
            self.wasted += 1
        loop = asyncio.get_running_loop()
        self.text = text
        self.task = asyncio.ensure_future(loop.run_in_executor(None, self.retriever.search, text))
        if self.on_result:
            self.task.add_done_callback(
                lambda t, text=text: t.cancelled() or t.exception() or self.on_result(text, t.result()))

    async def take(self, final: str):
        """(ranked, reused) for the final transcript."""
        task, text = self.task, self.text
        self.task = self.text = None
        if task and text and _norm(text) == _norm(final):
            return await task, True
        if task:
            task.cancel()
            self.wasted += 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retriever.search, final), False
