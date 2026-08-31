SHELL := /bin/bash

.PHONY: setup play arena zip gate test

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

test:
	uv run pytest

zip:
	uv run python -m harness.package --include encoding.py --include inference.py
	uv run python -m tools.checkzip

gate:
	uv run ruff check .
	uv run mypy
	uv run pytest
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
