from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asr import Signal, Utterances, Whisper
from .events import SAMPLE_RATE_IN, Session
from .llm import OllamaStream
from .retrieval import Retriever
from .tts import PiperTTS
from .turn import Turn

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="voice-agent-rt")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
deps: dict = {}


@app.on_event("startup")
def load_models() -> None:
    deps["whisper"] = Whisper()
    deps["retriever"] = Retriever()
    deps["llm"] = OllamaStream()
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
    partial_task: asyncio.Task | None = None
    await session.emit("session_started", sample_rate_in=SAMPLE_RATE_IN,
                       sample_rate_out=deps["tts"].sample_rate)

    async def partial(pcm: bytes) -> None:
        text = await loop.run_in_executor(None, whisper.transcribe, pcm)
        if text:
            await session.emit("partial_transcript", text=text)

    try:
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "audio":
                signal = utt.push(base64.b64decode(msg["pcm"]))
                if signal is Signal.PARTIAL_DUE and (partial_task is None or partial_task.done()):
                    partial_task = asyncio.create_task(partial(utt.audio()))
                elif signal is Signal.END:
                    text = await loop.run_in_executor(None, whisper.transcribe, utt.audio())
                    utt.reset()
                    await session.emit("final_transcript", text=text)
                    if text:
                        await Turn(session, deps["retriever"], deps["llm"], deps.get("tts")).run(text)
            elif msg["type"] == "end_session":
                break
    except WebSocketDisconnect:
        pass
