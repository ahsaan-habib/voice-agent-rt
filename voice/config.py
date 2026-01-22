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

# Primary model. Point LLM_URL at a bigger GPU box if you have one; the
# fallback stays on this machine so it can't fail for the same reason.
LLM_URL = env("LLM_URL", "http://localhost:11434")
LLM_MODEL = env("LLM_MODEL", "qwen3:4b")
FALLBACK_URL = env("FALLBACK_URL", "http://localhost:11434")
FALLBACK_MODEL = env("FALLBACK_MODEL", "llama3.2:3b")

# Every timeout is a promise about the worst case.
LLM_FIRST_TOKEN_S = float(env("LLM_FIRST_TOKEN_S", "4.0"))
LLM_STALL_S = float(env("LLM_STALL_S", "3.0"))       # max gap between tokens
ASR_TIMEOUT_S = float(env("ASR_TIMEOUT_S", "3.0"))
TTS_TIMEOUT_S = float(env("TTS_TIMEOUT_S", "5.0"))
RETRIEVAL_TIMEOUT_S = float(env("RETRIEVAL_TIMEOUT_S", "2.0"))

# bge-reranker logits; below this nothing retrieved is about the question
RETRIEVAL_MIN_SCORE = float(env("RETRIEVAL_MIN_SCORE", "-2.0"))
