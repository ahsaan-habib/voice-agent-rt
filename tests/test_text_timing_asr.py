import json
import sys

import pytest

from voice import asr, timing, waterfall
from voice.events import FRAME_BYTES
from voice.text import pop_sentence


def test_pop_sentence_waits_for_a_real_sentence_end():
    assert pop_sentence("Use version 3.2 of the package") == (None, "Use version 3.2 of the package")
    s, rest = pop_sentence("Use e.g. the queue helper for this. Then more")
    assert s == "Use e.g. the queue helper for this." and rest == " Then more"
    assert pop_sentence("Yes. It works fine here.")[0] == "Yes. It works fine here."


class ScriptedVAD:
    def __init__(self, pattern):
        self.pattern = iter(pattern)

    def is_speech(self, frame, rate):
        return next(self.pattern)


def push_all(utt, pattern):
    utt.vad = ScriptedVAD(pattern)
    return [utt.push(b"\x00" * FRAME_BYTES) for _ in pattern]


def test_utterance_start_partials_end():
    utt = asr.Utterances(silence_ms=60, partial_every_ms=100, min_speech_ms=100)
    sig = push_all(utt, [False, True] + [True] * 6 + [False] * 3)
    assert sig[0] is None and sig[1] is asr.Signal.SPEECH_START
    assert asr.Signal.PARTIAL_DUE in sig and sig[-1] is asr.Signal.END
    assert len(utt.audio()) == 10 * FRAME_BYTES


def test_a_cough_is_not_an_utterance():
    utt = asr.Utterances(silence_ms=60, partial_every_ms=1000, min_speech_ms=200)
    sig = push_all(utt, [True, True, False, False, False])
    assert asr.Signal.END not in sig and not utt.in_speech and utt.audio() == b""


def test_real_webrtcvad_accepts_20ms_frames():
    assert asr.Utterances().push(b"\x00" * FRAME_BYTES) is None     # silence


def test_turn_timer_and_waterfall(tmp_path, monkeypatch, capsys):
    log = tmp_path / "turns.jsonl"
    monkeypatch.setattr(timing, "TURNS_LOG", log)
    monkeypatch.setattr(waterfall, "TURNS_LOG", log)
    monkeypatch.setattr(sys, "argv", ["waterfall"])
    with pytest.raises(SystemExit, match="no turns"):
        waterfall.main()
    t = timing.TurnTimer()
    t.start("retrieval")
    t.end("retrieval")
    t.mark("first_audio")
    t.mark("first_audio")       # first one wins
    t.save(text="q", retrieval_reused=True)
    rec = json.loads(log.read_text())
    assert rec["spans"]["retrieval"][1] is not None and list(rec["marks"]) == ["first_audio"]
    monkeypatch.setattr(sys, "argv", ["waterfall", "--stats"])
    waterfall.main()
    out = capsys.readouterr().out
    assert "retrieval reused from partial: 100%" in out and "first audio" in out
