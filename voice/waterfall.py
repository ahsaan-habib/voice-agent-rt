"""Latency waterfall per turn, plus p50/p95 per stage.

    python -m voice.waterfall                 # last 10 turns
    python -m voice.waterfall --all --stats
"""
from __future__ import annotations

import argparse
import json

from .timing import TURNS_LOG

ORDER = ["asr_final", "retrieval", "llm", "tts_0", "tts_1", "tts_2"]
WIDTH = 60


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def draw(turn: dict, scale_ms: float) -> None:
    print(f"\n{turn.get('text', '')[:70]!r}")
    for name in ORDER + sorted(set(turn["spans"]) - set(ORDER)):
        if name not in turn["spans"]:
            continue
        a, b = turn["spans"][name]
        b = b if b is not None else a
        lo, hi = int(a / scale_ms * WIDTH), max(int(b / scale_ms * WIDTH), int(a / scale_ms * WIDTH) + 1)
        print(f"  {name:<14}{' ' * lo}{'█' * (hi - lo)}{' ' * (WIDTH - hi)} {a:7.0f} → {b:7.0f} ms")
    for name, t in turn["marks"].items():
        pos = min(int(t / scale_ms * WIDTH), WIDTH - 1)
        print(f"  {name:<14}{' ' * pos}▲ {t:.0f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    turns = [json.loads(l) for l in TURNS_LOG.read_text().splitlines() if l.strip()]
    shown = turns if args.all else turns[-args.n:]
    scale = max((t["marks"].get("first_audio") or max((s[1] or s[0]) for s in t["spans"].values()))
                for t in shown) or 1
    for t in shown:
        draw(t, scale)

    if args.stats:
        fa = [t["marks"]["first_audio"] for t in turns if "first_audio" in t["marks"]]
        # p95 first: that's the number a person forms an opinion from
        if fa:
            print(f"\nfirst audio  p95 {pct(fa, .95):.0f} ms   p50 {pct(fa, .5):.0f} ms   ({len(fa)} turns)")
        reused = [t.get("retrieval_reused") for t in turns if "retrieval_reused" in t]
        if reused:
            print(f"retrieval reused from partial: {sum(map(bool, reused)) / len(reused):.0%}")
        print(f"\n{len(turns)} turns")
        print(f"  {'stage':<14}{'p50':>8}{'p95':>8}   (duration ms)")
        for name in ORDER:
            d = [s[name][1] - s[name][0] for s in (t["spans"] for t in turns) if name in s and s[name][1]]
            if d:
                print(f"  {name:<14}{pct(d, .5):8.0f}{pct(d, .95):8.0f}")
        for mark in ("first_token", "first_sentence", "first_audio"):
            v = [t["marks"][mark] for t in turns if mark in t["marks"]]
            if v:
                print(f"  {mark:<14}{pct(v, .5):8.0f}{pct(v, .95):8.0f}   (from speech end)")


if __name__ == "__main__":
    main()
