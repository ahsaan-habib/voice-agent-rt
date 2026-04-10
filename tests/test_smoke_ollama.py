"""Streamed answer from a real local model, through the resilience wrapper.
Opt-in:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import asyncio
import os

import pytest

from conftest import CHUNK
from voice.llm import OllamaStream
from voice.resilience import ResilientLLM
from voice.text import pop_sentence

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")


def test_streams_a_short_spoken_answer(monkeypatch):
    from voice import config
    monkeypatch.setattr(config, "LLM_FIRST_TOKEN_S", 30.0)    # conftest shortens it; a cold model needs longer
    monkeypatch.setattr(config, "LLM_STALL_S", 10.0)
    events = []

    async def emit(t, **d):
        events.append(t)

    async def go():
        llm = ResilientLLM(OllamaStream(), OllamaStream())    # fallback = same model: no extra pull
        return [t async for t in llm.stream("How do I remove a global scope for one query?", [CHUNK], emit)]

    tokens = asyncio.run(go())
    text = "".join(tokens)
    print("\n", len(tokens), "tokens:", text)
    assert "degraded" not in events and len(tokens) > 3 and ("withoutglobalscope" in text.replace(" ", "").lower() or "global scope" in text.lower())
    assert "```" not in text and pop_sentence(text)[0]
