"""voice/resilience.py — a deadline, a fallback, and an announcement.

The client gets a `degraded` event whenever we fall back, and shows it. A
system quietly giving worse answers loses trust slowly; saying so costs
nothing and is the difference between a fallback and a lie.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from rag_grounded.ingest.chunker import Chunk

from . import config
from .llm import OllamaStream

log = logging.getLogger(__name__)

Emit = Callable[..., Awaitable[None]]


async def _with_deadlines(it: AsyncIterator[str], first_s: float, stall_s: float) -> AsyncIterator[str]:
    first = True
    while True:
        try:
            token = await asyncio.wait_for(anext(it), first_s if first else stall_s)
        except StopAsyncIteration:
            return
        first = False
        yield token


class ResilientLLM:
    def __init__(self, primary: OllamaStream | None = None, fallback: OllamaStream | None = None):
        self.primary = primary or OllamaStream()
        # local model from local-slm-bench: worse answers, but on this box and
        # shorter, because a fallback that rambles defeats the point
        self.fallback = fallback or OllamaStream(config.FALLBACK_URL, config.FALLBACK_MODEL, num_predict=120)

    async def stream(self, query: str, chunks: list[Chunk], emit: Emit) -> AsyncIterator[str]:
        sent_any = False
        try:
            async for token in _with_deadlines(self.primary.stream(query, chunks),
                                               config.LLM_FIRST_TOKEN_S, config.LLM_STALL_S):
                sent_any = True
                yield token
            return
        except (asyncio.TimeoutError, httpx.HTTPError) as e:
            log.warning("llm.fallback reason=%s after_tokens=%s", type(e).__name__, sent_any)
            reason = type(e).__name__

        await emit("degraded", stage="llm", reason=reason, model=self.fallback.model)
        if sent_any:
            # mid-answer failure: don't restart the answer from the top
            yield " Sorry, I lost my train of thought there."
            return
        try:
            async for token in _with_deadlines(self.fallback.stream(query, chunks),
                                               config.LLM_FIRST_TOKEN_S, config.LLM_STALL_S):
                yield token
        except (asyncio.TimeoutError, httpx.HTTPError) as e:
            # both down: say so and end the turn, rather than leave the client waiting
            log.warning("llm.fallback_failed reason=%s", type(e).__name__)
            await emit("degraded", stage="llm_fallback", reason=type(e).__name__)
            yield " Sorry, I can't answer right now. Please try again in a moment."
