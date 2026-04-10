"""Offline by default: no Whisper, no Piper, no Ollama, no index. Each stage
is a small fake so the turn logic, fallbacks and the socket protocol can be
exercised in milliseconds. test_smoke_ollama.py is opt-in."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
_tmp = tempfile.mkdtemp(prefix="voice-test-")
os.environ.setdefault("VOICE_RECORDINGS", os.path.join(_tmp, "recordings"))
os.environ.setdefault("VOICE_TURNS_LOG", os.path.join(_tmp, "turns.jsonl"))
# short deadlines so the timeout paths run fast
os.environ.setdefault("VOICE_LLM_FIRST_TOKEN_S", "0.3")
os.environ.setdefault("VOICE_LLM_STALL_S", "0.3")

import fake_st  # noqa: E402

fake_st.install()

from rag_grounded.ingest.chunker import Chunk  # noqa: E402

CHUNK = Chunk("c1", "laravel/eloquent.md", "Removing Global Scopes",
              "Use withoutGlobalScope to remove a global scope for one query.")


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)

    async def send_bytes(self, pcm):
        self.sent.append({"type": "audio_out", "bytes": len(pcm)})

    def types(self):
        return [m["type"] for m in self.sent]


class FakeRetriever:
    def __init__(self, ranked=None):
        self.ranked = [(CHUNK, 3.0)] if ranked is None else ranked
        self.queries = []

    def search(self, q):
        self.queries.append(q)
        return self.ranked

    def warm(self):
        pass


class FakeTTS:
    sample_rate = 22050

    def __init__(self, fail=False):
        self.fail, self.said = fail, []

    def synth(self, text):
        if self.fail:
            raise RuntimeError("voice broke")
        self.said.append(text)
        return b"\x00\x00" * 100


class FakeLLM:
    """Stands in for ResilientLLM.stream."""

    def __init__(self, tokens=("Use withoutGlobalScope ", "on the query. ", "That removes ", "it for one query.")):
        self.tokens = tokens

    async def stream(self, query, chunks, emit):
        for t in self.tokens:
            yield t
