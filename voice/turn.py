"""One conversational turn: final transcript -> retrieval -> streamed answer."""
from __future__ import annotations

import asyncio
import logging
import time

from . import config
from .events import Session
from .resilience import ResilientLLM
from .retrieval import Prefetch, Retriever
from .text import pop_sentence
from .timing import TurnTimer

log = logging.getLogger(__name__)
TTS_RETRY_AFTER_S = 60


class Turn:
    def __init__(self, session: Session, retriever: Retriever, llm: ResilientLLM, tts=None,
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

        # Speak sentence one while the model is still writing sentence two:
        # the tail of generation drops out of perceived latency.
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(sentences)) if self.tts and self.session.tts_ok() else None

        buf = ""
        tm.start("llm")
        async for token in self.llm.stream(text, chunks, self.session.emit):
            tm.mark("first_token")
            await self.session.emit("token", text=token)
            buf += token
            sentence, buf = pop_sentence(buf)
            if sentence:
                tm.mark("first_sentence")
                await sentences.put(sentence)
        tm.end("llm")
        if buf.strip():
            await sentences.put(buf.strip())
        await sentences.put(None)
        if speaker:
            await speaker
        tm.save(text=text, retrieval_reused=reused)
        await self.session.emit("turn_complete", timings=tm.summary())

    async def _speak(self, sentences: "asyncio.Queue[str | None]") -> None:
        loop = asyncio.get_running_loop()
        tm = self.timer
        n = 0
        while (sentence := await sentences.get()) is not None:
            name = f"tts_{n}"
            tm.start(name)
            try:
                pcm = await asyncio.wait_for(loop.run_in_executor(None, self.tts.synth, sentence),
                                             config.TTS_TIMEOUT_S)
            except Exception as e:  # timeout or a broken voice: the answer is already on screen
                tm.end(name)
                log.warning("tts.down reason=%s", type(e).__name__)
                self.session.tts_down_until = time.monotonic() + TTS_RETRY_AFTER_S
                await self.session.emit("degraded", stage="tts", reason=type(e).__name__)
                while await sentences.get() is not None:   # drain, keep the producer unblocked
                    pass
                return
            tm.end(name)
            await self.session.send_audio(pcm)
            tm.mark("first_audio")
            n += 1


async def say(session: Session, tts, text: str) -> None:
    """Something the agent says about itself (fallbacks, refusals). Spoken if
    TTS works, always shown."""
    await session.emit("notice", text=text)
    if tts and session.tts_ok():
        loop = asyncio.get_running_loop()
        try:
            pcm = await asyncio.wait_for(loop.run_in_executor(None, tts.synth, text), config.TTS_TIMEOUT_S)
            await session.send_audio(pcm)
        except Exception:
            session.tts_down_until = time.monotonic() + TTS_RETRY_AFTER_S
            await session.emit("degraded", stage="tts", reason="notice")
