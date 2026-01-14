"""Per-turn latency budget. Everything is measured from the moment VAD
decided the person stopped talking — that's when they start waiting.

You can't optimise "one and a half seconds". You can optimise a 420 ms TTFT
once you can see it next to everything else.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

TURNS_LOG = Path(os.environ.get("VOICE_TURNS_LOG", "turns.jsonl"))


class TurnTimer:
    def __init__(self, speech_end: float | None = None):
        self.t0 = speech_end or time.perf_counter()
        self.spans: dict[str, list[float]] = {}   # name -> [start_ms, end_ms]
        self.marks: dict[str, float] = {}

    def _ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def start(self, name: str) -> None:
        self.spans[name] = [self._ms(), None]

    def end(self, name: str) -> None:
        if name in self.spans and self.spans[name][1] is None:
            self.spans[name][1] = self._ms()

    def mark(self, name: str) -> None:
        """First occurrence wins: first_token, first_audio..."""
        self.marks.setdefault(name, self._ms())

    def summary(self) -> dict:
        return {"spans": self.spans, "marks": self.marks}

    def save(self, **extra) -> None:
        with TURNS_LOG.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **self.summary(), **extra}) + "\n")
