"""Record a session: every inbound audio frame and typed message, every
outbound event, with timestamps. A recording is a fixture — "it did something
weird yesterday" becomes something you can run again.

recordings/<id>/
  audio.pcm     raw inbound s16le 16 kHz, frames concatenated
  inbound.jsonl {"t", "kind": "audio", "offset", "len"} | {"t", "kind": "text", "text"}
  events.jsonl  outbound events as sent
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

RECORDINGS = Path(os.environ.get("VOICE_RECORDINGS", "recordings"))
ENABLED = os.environ.get("VOICE_RECORD", "1") == "1"


class Recorder:
    def __init__(self, session_id: str | None = None):
        self.id = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = RECORDINGS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.audio = open(self.dir / "audio.pcm", "ab")
        self.inbound = open(self.dir / "inbound.jsonl", "a")
        self.events = open(self.dir / "events.jsonl", "a")
        self.offset = 0

    def _t(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def audio_frame(self, pcm: bytes) -> None:
        self.audio.write(pcm)
        self.inbound.write(json.dumps({"t": self._t(), "kind": "audio", "offset": self.offset, "len": len(pcm)}) + "\n")
        self.offset += len(pcm)

    def text(self, text: str) -> None:
        self.inbound.write(json.dumps({"t": self._t(), "kind": "text", "text": text}) + "\n")

    def event(self, msg: dict) -> None:
        self.events.write(json.dumps(msg) + "\n")

    def close(self) -> None:
        for f in (self.audio, self.inbound, self.events):
            f.close()
