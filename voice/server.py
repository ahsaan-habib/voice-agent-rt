from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asr import Signal, Utterances, Whisper
from .events import SAMPLE_RATE_IN, Session
from .resilience import ResilientLLM
from .retrieval import Prefetch, Retriever
from .timing import TurnTimer
from .tts import PiperTTS
from . import config
from .turn import Turn, say

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="voice-agent-rt")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
deps: dict = {}


@app.on_event("startup")
def load_models() -> None:
    deps["whisper"] = Whisper()
    deps["retriever"] = Retriever()
    deps["retriever"].warm()
    deps["llm"] = ResilientLLM()
    deps["tts"] = PiperTTS()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    whisper = deps["whisper"]
    session = Session(ws)
    utt = Utterances()
    prefetch = Prefetch(deps["retriever"])
    partial_task: asyncio.Task | None = None
    await session.emit("session_started", sample_rate_in=SAMPLE_RATE_IN,
                       sample_rate_out=deps["tts"].sample_rate)

    async def partial(pcm: bytes) -> None:
        try:
            text = await asyncio.wait_for(loop.run_in_executor(None, whisper.transcribe, pcm),
                                          config.ASR_TIMEOUT_S)
        except Exception:
            return   # the final transcription decides whether ASR is down
        if text:
            await session.emit("partial_transcript", text=text)
            prefetch.on_partial(text)

    async def asr_down(reason: str) -> None:
        session.asr_down_until = time.monotonic() + 30
        utt.reset()
        await session.emit("degraded", stage="asr", reason=reason, fallback="typed_input")
        await say(session, deps["tts"], "I'm having trouble hearing you. You can type your question instead.")

    async def turn(text: str, timer: TurnTimer) -> None:
        await Turn(session, deps["retriever"], deps["llm"], deps.get("tts"), timer).run(text, prefetch)

    try:
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "text_input":
                text = msg.get("text", "").strip()
                if text:
                    await session.emit("final_transcript", text=text, source="typed")
                    await turn(text, TurnTimer())
            elif msg["type"] == "audio":
                if not session.asr_ok():
                    continue
                signal = utt.push(base64.b64decode(msg["pcm"]))
                if signal is Signal.PARTIAL_DUE and (partial_task is None or partial_task.done()):
                    partial_task = asyncio.create_task(partial(utt.audio()))
                elif signal is Signal.END:
                    timer = TurnTimer(speech_end=time.perf_counter())
                    timer.start("asr_final")
                    try:
                        text = await asyncio.wait_for(
                            loop.run_in_executor(None, whisper.transcribe, utt.audio()), config.ASR_TIMEOUT_S)
                    except Exception as e:
                        await asr_down(type(e).__name__)
                        continue
                    timer.end("asr_final")
                    utt.reset()
                    await session.emit("final_transcript", text=text)
                    if text:
                        await turn(text, timer)
            elif msg["type"] == "end_session":
                break
    except WebSocketDisconnect:
        pass
