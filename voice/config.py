from __future__ import annotations

import os


def env(name: str, default: str) -> str:
    return os.environ.get(f"VOICE_{name}", default)


ASR_MODEL = env("ASR_MODEL", "base.en")          # faster-whisper size
ASR_DEVICE = env("ASR_DEVICE", "auto")
VAD_AGGRESSIVENESS = int(env("VAD_AGGRESSIVENESS", "2"))
# The silence that counts as "they stopped talking". A UX dial disguised as a
# latency number: shorter interrupts people, longer feels slow.
VAD_SILENCE_MS = int(env("VAD_SILENCE_MS", "240"))
PARTIAL_EVERY_MS = int(env("PARTIAL_EVERY_MS", "400"))

LLM_URL = env("LLM_URL", "http://localhost:11434")
LLM_MODEL = env("LLM_MODEL", "qwen3:4b")
