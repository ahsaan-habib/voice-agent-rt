"""One conversational turn: final transcript -> retrieval -> streamed answer."""
from __future__ import annotations

import asyncio

from .events import Session
from .llm import OllamaStream
from .retrieval import Prefetch, Retriever
from .timing import TurnTimer


class Turn:
    def __init__(self, session: Session, retriever: Retriever, llm: OllamaStream, tts=None,
                 timer: TurnTimer | None = None):
        self.timer = timer or TurnTimer()
        self.session = session
        self.retriever = retriever
        self.llm = llm
        self.tts = tts

    async def run(self, text: str, prefetch: Prefetch | None = None) -> None:
        loop = asyncio.get_running_loop()
        tm = self.timer
        tm.start("retrieval")
        if prefetch:
            ranked, reused = await prefetch.take(text)
        else:
            ranked, reused = await loop.run_in_executor(None, self.retriever.search, text), False
        tm.end("retrieval")
        chunks = [c for c, _ in ranked]
        answer = []
        tm.start("llm")
        async for token in self.llm.stream(text, chunks):
            tm.mark("first_token")
            answer.append(token)
            await self.session.emit("token", text=token)
        tm.end("llm")
        if self.tts:
            tm.start("tts")
            pcm = await loop.run_in_executor(None, self.tts.synth, "".join(answer))
            tm.end("tts")
            await self.session.send_audio(pcm)
            tm.mark("first_audio")
        tm.save(text=text, retrieval_reused=reused)
        await self.session.emit("turn_complete", timings=tm.summary())
