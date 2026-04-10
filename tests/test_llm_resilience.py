import asyncio
import json

import httpx
import pytest

from conftest import CHUNK
from voice import config
from voice.llm import OllamaStream, build_messages
from voice.resilience import ResilientLLM


def test_build_messages_carries_sources():
    m = build_messages("how?", [CHUNK])
    assert "(laravel/eloquent.md — Removing Global Scopes)" in m[1]["content"] and "how?" in m[1]["content"]


def test_ollama_stream_parses_ndjson():
    def handler(req):
        body = json.loads(req.content)
        assert body["stream"] and body["options"]["seed"] == config.LLM_SEED
        lines = [{"message": {"content": "Hi"}, "done": False}, {"message": {"content": " there"}, "done": False},
                 {"done": True}]
        return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines).encode())

    s = OllamaStream(base_url="http://x")
    s.http = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))

    async def collect():
        return [t async for t in s.stream("q", [CHUNK])]

    assert asyncio.run(collect()) == ["Hi", " there"]


class Stream:
    def __init__(self, tokens=(), delay_first=0.0, fail_after=None, model="m"):
        self.tokens, self.delay_first, self.fail_after, self.model = tokens, delay_first, fail_after, model

    async def stream(self, query, chunks):
        await asyncio.sleep(self.delay_first)
        for i, t in enumerate(self.tokens):
            if self.fail_after is not None and i == self.fail_after:
                raise httpx.ReadError("connection dropped")
            yield t


def run(llm):
    events = []

    async def emit(type_, **data):
        events.append((type_, data))

    async def go():
        return [t async for t in llm.stream("q", [CHUNK], emit)]

    return asyncio.run(go()), events


def test_primary_ok_no_fallback():
    toks, events = run(ResilientLLM(Stream(["a", "b"]), Stream(["fallback"])))
    assert toks == ["a", "b"] and events == []


def test_slow_first_token_falls_back_and_says_so():
    toks, events = run(ResilientLLM(Stream(["late"], delay_first=1.0), Stream(["quick"], model="small")))
    assert toks == ["quick"]
    assert events == [("degraded", {"stage": "llm", "reason": "TimeoutError", "model": "small"})]


def test_failure_mid_answer_does_not_restart_it():
    toks, events = run(ResilientLLM(Stream(["one ", "two ", "three"], fail_after=2), Stream(["fallback"])))
    assert toks[:2] == ["one ", "two "] and "lost my train of thought" in toks[2] and len(toks) == 3
    assert events[0][1]["reason"] == "ReadError"


def test_both_models_down_still_ends_the_answer():
    toks, events = run(ResilientLLM(Stream(["x"], delay_first=1.0), Stream(["y"], delay_first=1.0)))
    assert toks == [" Sorry, I can't answer right now. Please try again in a moment."]
    assert [e[1]["stage"] for e in events] == ["llm", "llm_fallback"]
