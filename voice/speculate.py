"""Start generating on the partial transcript too. If the final transcript
matches, the time to first token is already paid."""
from __future__ import annotations

import asyncio

from .llm import OllamaStream
from .retrieval import _norm


class SpeculativeLLM:
    def __init__(self, llm: OllamaStream):
        self.llm = llm
        self.text: str | None = None
        self.tokens: list[str] = []
        self.task: asyncio.Task | None = None
        self.done = asyncio.Event()

    def start(self, text: str, chunks) -> None:
        self.cancel()
        self.text, self.tokens, self.done = text, [], asyncio.Event()

        async def run():
            async for t in self.llm.stream(text, chunks):
                self.tokens.append(t)
            self.done.set()

        self.task = asyncio.create_task(run())

    def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    async def take(self, final: str):
        """Async iterator of tokens if the speculation matches, else None."""
        if not self.task or _norm(self.text or "") != _norm(final):
            self.cancel()
            return None

        async def replay():
            i = 0
            while True:
                while i < len(self.tokens):
                    yield self.tokens[i]
                    i += 1
                if self.done.is_set() and i >= len(self.tokens):
                    return
                await asyncio.sleep(0.005)

        return replay()
