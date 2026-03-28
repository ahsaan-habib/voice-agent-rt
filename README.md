# voice-agent-rt

A voice interface to the Laravel / Filament / Livewire docs, running entirely
on one machine: browser mic → VAD → ASR → hybrid retrieval → LLM → TTS →
speaker, over a single WebSocket.

The interesting parts aren't the components, they're the **latency budget**
(where every millisecond between "they stopped talking" and "it started
speaking" goes) and **what happens when a component fails** (never silence).

| Stage | Component |
|---|---|
| VAD | webrtcvad, 240 ms of silence ends an utterance |
| ASR | faster-whisper `base.en`, partials by re-transcribing the buffer every 400 ms |
| Retrieval | [rag-grounded](https://github.com/ahsaan-habib/rag-grounded) hybrid BM25 + dense, RRF, cross-encoder rerank |
| LLM | Ollama `qwen3:4b` (primary), `llama3.2:3b` (fallback) |
| TTS | Piper `en_US-lessac-medium` |

## Protocol

One WebSocket. Audio is binary frames both ways (raw s16le); everything else
is typed JSON events (see `voice/events.py`): `partial_transcript`,
`final_transcript`, `token`, `degraded`, `notice`, `refusal`,
`turn_complete {timings}`. Every optimisation below is a rearrangement of when
these fire.

## Latency budget

Every turn records spans and marks relative to *speech end* (the moment VAD
decides you stopped) into `turns.jsonl`:

```
python -m voice.waterfall -n 5 --stats

# illustrative output — shape of the view, not a measurement
'how do I stop a global scope applying to one query'
  asr_final     ███                                     0 →    95 ms
  retrieval     █                                       96 →   104 ms     (prefetched on the partial)
  llm             ██████████████████████████████       104 →  1890 ms
  tts_0                   ████                         640 →   890 ms
  first_token       ▲ 520 ms
  first_audio                 ▲ 890 ms
```

What changed the shape (each is one commit):

1. **Retrieve on the partial transcript.** Retrieval starts while ASR is still
   finalising; reused if the final transcript matches, discarded otherwise.
2. **Sentence-level TTS.** Sentence one is synthesised while the model writes
   sentence two, so generation's tail leaves the perceived latency.
3. **Warm the reranker** at startup, and rerank 20 candidates instead of 30.
   A cold cross-encoder penalises exactly the first request after idle.
4. **Binary audio frames** instead of base64 inside JSON. Less pleasant to
   read in devtools, less work on every 20 ms frame.

And one that was reverted: starting *generation* on the partial transcript.
It cut p50 a little but wasted tokens on every mismatched partial and could
leak a cancelled generation's tokens to the client. Speculating retrieval is
cheap; speculating generation isn't.

## Degradation

| Failure | Detected by | What happens |
|---|---|---|
| ASR down / slow | exception or 3 s timeout | switch to typed input, say so out loud |
| Primary LLM | 4 s to first token, 3 s stall between tokens | fall back to `llama3.2:3b`, `degraded` event, shorter answer |
| TTS down / slow | exception or 5 s timeout | answer continues as text; voice retried after 60 s |
| Retrieval empty / slow | no chunk over the rerank threshold, or 2 s timeout | spoken refusal, offer to rephrase |

The client shows a badge for every `degraded` event. Quietly serving worse
answers is how a fallback becomes a lie.

`VOICE_LLM_URL` can point the primary at a bigger GPU box while the fallback
stays on localhost, so they can't fail for the same reason.

## Replay

Every session is recorded under `recordings/<id>/` (inbound audio frames +
typed input + outbound events). Replay feeds them back through a running
server and diffs transcripts, answers, degradations and first-audio latency:

```bash
python -m voice.replay recordings/ --speed 2
```

Use it to check that a latency change helps on real conversations, not just
on the clearest sentence you can say into your laptop.

## Run it

```bash
make install && source .venv/bin/activate
make models          # ollama pulls + piper voice
# rag-grounded index: RAG_INDEX_DIR=../rag-grounded/.chroma
make serve           # http://localhost:8000
```

## Not solved

Barge-in (talking over the agent) isn't handled yet. One language, quiet
room. Single session — the warm reranker is shared and nothing here has been
measured under concurrency.
