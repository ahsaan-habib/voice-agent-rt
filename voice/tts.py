"""Piper TTS (local, onnx). Download a voice first:
    python -m voice.tts download en_US-lessac-medium
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

VOICES_DIR = Path("voices")
DEFAULT_VOICE = "en_US-lessac-medium"


class PiperTTS:
    def __init__(self, voice: str = DEFAULT_VOICE):
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(VOICES_DIR / f"{voice}.onnx"))
        self.sample_rate = self.voice.config.sample_rate

    def synth(self, text: str) -> bytes:
        """s16le mono PCM at self.sample_rate."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize_wav(text, wav)
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            return wav.readframes(wav.getnframes())


def download(voice: str = DEFAULT_VOICE) -> None:
    import httpx

    lang, name, quality = voice.split("-")
    base = (f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}")
    VOICES_DIR.mkdir(exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        out = VOICES_DIR / f"{voice}{ext}"
        if not out.exists():
            out.write_bytes(httpx.get(base + ext, follow_redirects=True, timeout=120).content)
            print(f"-> {out}")


if __name__ == "__main__" and sys.argv[1:2] == ["download"]:
    download(*sys.argv[2:3])
