import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import CHUNK, FakeLLM, FakeRetriever, FakeTTS, FakeWS
from voice import recorder, replay, server, timing
from voice.events import Session
from voice.retrieval import Prefetch
from voice.turn import Turn


def run_turn(retriever=None, llm=None, tts=None, prefetch=False, text="how do I remove a global scope?"):
    ws = FakeWS()
    session = Session(ws)
    retriever = retriever or FakeRetriever()

    async def go():
        pf = Prefetch(retriever) if prefetch else None
        await Turn(session, retriever, llm or FakeLLM(), tts).run(text, pf)

    asyncio.run(go())
    return ws


def test_answer_is_spoken_sentence_by_sentence():
    tts = FakeTTS()
    ws = run_turn(tts=tts)
    assert ws.types().count("token") == 4 and ws.types()[-1] == "turn_complete"
    assert tts.said == ["Use withoutGlobalScope on the query.", "That removes it for one query."]
    marks = ws.sent[-1]["timings"]["marks"]
    assert {"first_token", "first_sentence", "first_audio"} <= marks.keys()


def test_nothing_retrieved_means_refusal_not_improvisation():
    ws = run_turn(retriever=FakeRetriever(ranked=[(CHUNK, -9.0)]), tts=FakeTTS())
    assert "refusal" in ws.types() and "token" not in ws.types() and ws.types()[-1] == "turn_complete"


def test_broken_tts_degrades_but_text_still_arrives():
    ws = run_turn(tts=FakeTTS(fail=True))
    assert ("degraded" in ws.types() and ws.types().count("token") == 4
            and next(m for m in ws.sent if m["type"] == "degraded")["stage"] == "tts")


def test_prefetch_reuses_a_matching_partial():
    retriever = FakeRetriever()

    async def go():
        pf = Prefetch(retriever, min_words=2)
        pf.on_partial("remove a global scope")
        ranked, reused = await pf.take("Remove a global scope!")
        pf.on_partial("something else entirely")
        _, reused2 = await pf.take("a different final")
        return reused, reused2, pf.wasted

    reused, reused2, wasted = asyncio.run(go())
    assert reused is True and reused2 is False and wasted == 1
    assert retriever.queries == ["remove a global scope", "something else entirely", "a different final"]


def test_recorder_keeps_session_ids_inside_recordings(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "RECORDINGS", tmp_path / "recordings")
    r = recorder.Recorder("../../escape")
    assert r.dir.resolve().parent == (tmp_path / "recordings").resolve() and r.id == "escape"
    r.close()
    assert recorder.Recorder("replay-20260101-abc").id == "replay-20260101-abc"


def test_websocket_typed_turn_and_replay_grouping(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "TURNS_LOG", tmp_path / "turns.jsonl")
    monkeypatch.setattr(recorder, "RECORDINGS", tmp_path / "recordings")
    monkeypatch.setattr(server, "RECORDING", True)
    monkeypatch.setattr(server, "Whisper", lambda: object())
    monkeypatch.setattr(server, "Retriever", FakeRetriever)
    monkeypatch.setattr(server, "ResilientLLM", FakeLLM)
    monkeypatch.setattr(server, "PiperTTS", FakeTTS)
    with TestClient(server.app) as client, client.websocket_connect("/ws?session=t1") as ws:
        started = ws.receive_json()
        assert started["type"] == "session_started" and started["sample_rate_out"] == 22050
        ws.send_text(json.dumps({"type": "text_input", "text": "how do I remove a global scope?"}))
        got = []
        while not got or got[-1].get("type") != "turn_complete":
            msg = ws.receive()
            got.append({"type": "audio_out"} if msg.get("bytes") else json.loads(msg["text"]))
        ws.send_text(json.dumps({"type": "end_session"}))
    types = [m["type"] for m in got]
    assert types[0] == "final_transcript" and "audio_out" in types
    rec = Path(tmp_path / "recordings" / "t1")
    turns = replay.turns(replay.load(rec)[2])
    assert turns[0]["transcript"].startswith("how do I") and "withoutGlobalScope" in turns[0]["answer"]
    assert replay.compare(rec, [json.loads(l) for l in (rec / "events.jsonl").read_text().splitlines()])
