"""Replay recorded sessions through a running server and compare.

    python -m voice.replay recordings/20260123-212210-a1b2c3
    python -m voice.replay recordings/ --speed 4        # every session, 4x pace

Feeds the recorded frames with their original pacing (or faster), collects the
events, and reports per turn: did the transcript, the answer and the
degradations match the recording, and how did first-audio latency move.
Pair it with a fixed LLM seed (VOICE_LLM_SEED) for deterministic answers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


def load(rec: Path) -> tuple[list[dict], bytes, list[dict]]:
    inbound = [json.loads(l) for l in (rec / "inbound.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (rec / "events.jsonl").read_text().splitlines() if l.strip()]
    return inbound, (rec / "audio.pcm").read_bytes(), events


def turns(events: list[dict]) -> list[dict]:
    """Group events into turns: transcript, answer text, degradations, first-audio latency."""
    out, cur = [], None
    for e in events:
        if e["type"] == "final_transcript":
            cur = {"transcript": e["text"], "answer": "", "degraded": [], "refused": False}
            out.append(cur)
        elif cur is None:
            continue
        elif e["type"] == "token":
            cur["answer"] += e["text"]
        elif e["type"] == "degraded":
            cur["degraded"].append(e["stage"])
        elif e["type"] == "refusal":
            cur["refused"] = True
        elif e["type"] == "turn_complete" and e.get("timings"):
            cur["first_audio_ms"] = e["timings"]["marks"].get("first_audio")
    return out


async def run(rec: Path, url: str, speed: float) -> list[dict]:
    inbound, audio, _ = load(rec)
    got: list[dict] = []
    async with websockets.connect(f"{url}?session=replay-{rec.name}", max_size=None) as ws:
        async def reader():
            async for raw in ws:
                if isinstance(raw, bytes):
                    got.append({"type": "audio_out", "bytes": len(raw)})
                else:
                    got.append(json.loads(raw))

        reader_task = asyncio.create_task(reader())
        last_t = 0.0
        for item in inbound:
            await asyncio.sleep(max(0.0, (item["t"] - last_t) / 1000 / speed))
            last_t = item["t"]
            if item["kind"] == "audio":
                pcm = audio[item["offset"]: item["offset"] + item["len"]]
                await ws.send(pcm)
            else:
                await ws.send(json.dumps({"type": "text_input", "text": item["text"]}))
        # let the last turn finish
        for _ in range(300):
            if got and got[-1]["type"] == "turn_complete":
                break
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "end_session"}))
        reader_task.cancel()
    return got


def compare(rec: Path, replayed: list[dict]) -> bool:
    _, _, original = load(rec)
    a, b = turns(original), turns(replayed)
    ok = len(a) == len(b)
    print(f"\n{rec.name}: {len(a)} turns recorded, {len(b)} replayed")
    for i, (x, y) in enumerate(zip(a, b), 1):
        same_t, same_a = x["transcript"] == y["transcript"], x["answer"] == y["answer"]
        ok &= same_t and same_a and x["degraded"] == y["degraded"]
        fa_x, fa_y = x.get("first_audio_ms"), y.get("first_audio_ms")
        delta = f"{fa_y - fa_x:+.0f} ms" if fa_x is not None and fa_y is not None else "n/a"
        print(f"  turn {i}: transcript {'=' if same_t else '≠'}  answer {'=' if same_a else '≠'}  "
              f"degraded {x['degraded']}→{y['degraded']}  first audio {fa_x}→{fa_y} ({delta})")
        if not same_t:
            print(f"    was: {x['transcript']!r}\n    now: {y['transcript']!r}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="a recording dir, or a dir of recordings")
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--speed", type=float, default=1.0, help="pacing multiplier (1 = real time)")
    args = ap.parse_args()

    recs = [args.path] if (args.path / "inbound.jsonl").exists() else sorted(
        p for p in args.path.iterdir() if (p / "inbound.jsonl").exists() and not p.name.startswith("replay-"))
    results = [compare(r, asyncio.run(run(r, args.url, args.speed))) for r in recs]
    print(f"\n{sum(results)}/{len(results)} sessions replayed identically")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
