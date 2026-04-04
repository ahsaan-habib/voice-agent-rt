"""The socket protocol. Every later optimisation is a rearrangement of when
these fire, so this is the part to get right first.

Audio travels as binary frames in both directions (raw s16le mono); every
other message is a JSON text frame. Base64-in-JSON cost ~33% more bytes plus
encode/decode on every 20 ms frame.

client -> server
  <binary>          20 ms of s16le 16 kHz mono mic audio
  {"type": "text_input", "text": "..."}          typed fallback
  {"type": "end_session"}

server -> client
  <binary>          TTS audio, s16le mono at sample_rate_out
  session_started   {sample_rate_in, sample_rate_out}
  partial_transcript{text}
  final_transcript  {text}
  token             {text}
  degraded          {stage, reason}       a fallback is in use — the client shows it
  notice            {text}                something the agent says about itself
  refusal           {reason}              nothing retrieved to ground an answer
  interrupted       {}                    user spoke over the answer; drop queued audio
  turn_complete     {timings}
  error             {message}
"""
from __future__ import annotations

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
        self.tts_down_until = 0.0
        self.asr_down_until = 0.0

    def tts_ok(self) -> bool:
        return time.monotonic() >= self.tts_down_until

    def asr_ok(self) -> bool:
        return time.monotonic() >= self.asr_down_until

    async def emit(self, type_: str, **data: Any) -> None:
        msg = {"type": type_, "t": round((time.perf_counter() - self.started) * 1000, 1), **data}
        if self.recorder:
            self.recorder.event(msg)
        await self.ws.send_json(msg)

    async def send_audio(self, pcm: bytes) -> None:
        if self.recorder:
            self.recorder.event({"type": "audio_out", "t": round((time.perf_counter() - self.started) * 1000, 1),
                                 "bytes": len(pcm)})
        await self.ws.send_bytes(pcm)
