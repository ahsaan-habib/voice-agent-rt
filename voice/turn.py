"""One conversational turn: final transcript -> retrieval -> streamed answer."""
from __future__ import annotations

import asyncio

from .events import Session
from .llm import OllamaStream
from .retrieval import Retriever


class Turn:
    def __init__(self, session: Session, retriever: Retriever, llm: OllamaStream, tts=None):
        self.session = session
        self.retriever = retriever
        self.llm = llm
        self.tts = tts

    async def run(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        ranked = await loop.run_in_executor(None, self.retriever.search, text)
        chunks = [c for c, _ in ranked]
        answer = []
        async for token in self.llm.stream(text, chunks):
            answer.append(token)
            await self.session.emit("token", text=token)
        if self.tts:
            pcm = await loop.run_in_executor(None, self.tts.synth, "".join(answer))
            await self.session.send_audio(pcm)
        await self.session.emit("turn_complete")
