"""The submission entrypoint. The platform imports this file and calls get_move.

This version has no tree search yet: it scores every legal move by the value the
network gives the resulting position (a one-ply look-ahead) and uses the policy
prior only to break near-ties. It exists to exercise the encoding -> ONNX -> move
path end to end and to measure the raw net against the baselines. PUCT search
lands on top of the same Evaluator next.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess

from inference import Evaluator

_MODEL_PATH = Path(__file__).with_name("weights") / "model.onnx"
_PRIOR_WEIGHT = 0.15

# Loaded once per game, inside the 60 s import budget, before the clock starts.
try:
    _evaluator: Evaluator | None = Evaluator(_MODEL_PATH)
except Exception as error:  # missing or broken weights: still play legal moves
    print(f"evaluator unavailable, falling back to random: {error}")
    _evaluator = None


def _best_move(board: chess.Board, legal: list[chess.Move]) -> chess.Move:
    evaluator = _evaluator
    if evaluator is None:
        return random.choice(legal)

    priors, _ = evaluator.evaluate([board])[0]

    children: list[chess.Board] = []
    for move in legal:
        board.push(move)
        children.append(board.copy(stack=False))
        board.pop()
    child_values = [value for _, value in evaluator.evaluate(children)]

    # child_value is from the opponent's point of view, so our score is its negation.
    best = legal[0]
    best_score = -1e9
    for move, child_value in zip(legal, child_values, strict=True):
        score = -child_value + _PRIOR_WEIGHT * priors.get(move, 0.0)
        if score > best_score:
            best_score = score
            best = move
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()
    try:
        return _best_move(board, legal).uci()
    except Exception as error:  # never forfeit on a bug; play a legal move
        print(f"get_move fell back to random: {error}")
        return random.choice(legal).uci()
