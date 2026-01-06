from __future__ import annotations

import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .events import SAMPLE_RATE_IN, Session

app = FastAPI(title="voice-agent-rt")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    await session.emit("session_started", sample_rate_in=SAMPLE_RATE_IN)
    try:
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "audio":
                pcm = base64.b64decode(msg["pcm"])
                await session.emit("partial_transcript", text=f"<{len(pcm)} bytes>")
            elif msg["type"] == "end_session":
                break
    except WebSocketDisconnect:
        pass
