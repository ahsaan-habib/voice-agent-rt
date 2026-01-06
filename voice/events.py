"""The socket protocol. Every later optimisation is a rearrangement of when
these fire, so this is the part to get right first.

client -> server
  {"type": "audio", "pcm": <base64 s16le 16 kHz mono, 20 ms>}
  {"type": "text_input", "text": "..."}          typed fallback
  {"type": "end_session"}

server -> client
  session_started   {sample_rate_in, sample_rate_out}
  partial_transcript{text}
  final_transcript  {text}
  token             {text}
  audio_chunk       {pcm: <base64 s16le>}
  turn_complete     {timings}
  error             {message}
"""
from __future__ import annotations

import base64
import time
from typing import Any

from fastapi import WebSocket

SAMPLE_RATE_IN = 16_000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE_IN * FRAME_MS // 1000 * 2   # 640


class Session:
    def __init__(self, ws: WebSocket, recorder=None):
        self.ws = ws
        self.started = time.perf_counter()
        self.recorder = recorder

    async def emit(self, type_: str, **data: Any) -> None:
        msg = {"type": type_, "t": round((time.perf_counter() - self.started) * 1000, 1), **data}
        if self.recorder:
            self.recorder.event(msg)
        await self.ws.send_json(msg)

    async def send_audio(self, pcm: bytes) -> None:
        await self.emit("audio_chunk", pcm=base64.b64encode(pcm).decode())
