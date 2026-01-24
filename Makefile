.PHONY: install models serve waterfall replay

install:
	python -m venv .venv && .venv/bin/pip install -e .

models:
	ollama pull qwen3:4b
	ollama pull llama3.2:3b
	python -m voice.tts download en_US-lessac-medium

serve:
	uvicorn voice.server:app --port 8000

waterfall:
	python -m voice.waterfall -n 5 --stats

replay:
	python -m voice.replay recordings/ --speed 2
