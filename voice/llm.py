from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag_grounded.ingest.chunker import Chunk

from . import config

SYSTEM = """You are a voice assistant for Laravel, Filament and Livewire
developers. Answer ONLY from the context passages. Your answer will be spoken
aloud: two to four short sentences, no markdown, no lists, no code blocks —
describe code in words (say "call without global scope" rather than writing it).
If the passages don't answer the question, say you don't have anything on that."""


def build_messages(query: str, chunks: list[Chunk]) -> list[dict]:
    context = "\n\n".join(f"({c.source} — {c.heading})\n{c.text}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


class OllamaStream:
    def __init__(self, base_url: str = config.LLM_URL, model: str = config.LLM_MODEL,
                 num_predict: int = 200):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(30.0, connect=2.0))
        self.model = model
        self.num_predict = num_predict

    async def stream(self, query: str, chunks: list[Chunk]) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": build_messages(query, chunks),
            "stream": True,
            "think": False,
            "keep_alive": "30m",   # same reason as the warm reranker: no cold model mid-conversation
            "options": {"temperature": 0.0, "num_predict": self.num_predict},
        }
        async with self.http.stream("POST", "/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    return
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
