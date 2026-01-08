"""VAD-segmented utterances + faster-whisper.

Whisper isn't a streaming model, so partials are produced by re-transcribing
the utterance-so-far every PARTIAL_EVERY_MS. Wasteful, but on base.en a
2-second buffer is ~100 ms on CPU and partials are what make everything after
this able to start early.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import webrtcvad

from . import config
from .events import FRAME_MS, SAMPLE_RATE_IN


class Signal(Enum):
    SPEECH_START = "speech_start"
    PARTIAL_DUE = "partial_due"
    END = "end"


class Utterances:
    """Feed 20 ms frames, get told when speech starts, when a partial is due,
    and when the utterance has ended."""

    def __init__(self, silence_ms: int = config.VAD_SILENCE_MS,
                 partial_every_ms: int = config.PARTIAL_EVERY_MS,
                 min_speech_ms: int = 200):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.silence_frames = silence_ms // FRAME_MS
        self.partial_frames = partial_every_ms // FRAME_MS
        self.min_speech_frames = min_speech_ms // FRAME_MS
        self.reset()

    def reset(self) -> None:
        self.buf = bytearray()
        self.in_speech = False
        self.speech_frames = 0
        self.trailing_silence = 0
        self.since_partial = 0

    def push(self, frame: bytes) -> Signal | None:
        speech = self.vad.is_speech(frame, SAMPLE_RATE_IN)
        if not self.in_speech:
            if not speech:
                return None
            self.in_speech = True
            self.buf.extend(frame)
            self.speech_frames = 1
            return Signal.SPEECH_START

        self.buf.extend(frame)
        if speech:
            self.speech_frames += 1
            self.trailing_silence = 0
        else:
            self.trailing_silence += 1
            if self.trailing_silence >= self.silence_frames:
                if self.speech_frames < self.min_speech_frames:
                    self.reset()      # a cough, a click — not an utterance
                    return None
                return Signal.END
        self.since_partial += 1
        if self.since_partial >= self.partial_frames:
            self.since_partial = 0
            return Signal.PARTIAL_DUE
        return None

    def audio(self) -> bytes:
        return bytes(self.buf)


class Whisper:
    def __init__(self, model: str = config.ASR_MODEL, device: str = config.ASR_DEVICE):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device,
                                  compute_type="int8" if device in ("cpu", "auto") else "float16")

    def transcribe(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1,
                                            vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
