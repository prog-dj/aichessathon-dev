"""The submission entrypoint. The platform imports this file and calls get_move.

A PUCT search (search.py) guided by the network (inference.py). This file owns
the two things the search needs from outside the position: a running board that
carries the game's move history (so repetitions and the fifty-move rule are
visible in the tree), and a time budget for the current move.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import chess

from inference import Evaluator
from search import PuctSearch

_MODEL_PATH = Path(__file__).with_name("weights") / "model.onnx"
_SAFETY_MS = 500  # never plan to use the last half second of the clock
_PANIC_MS = 4_000

try:
    _search: PuctSearch | None = PuctSearch(Evaluator(_MODEL_PATH))
except Exception as error:  # missing or broken weights: still play legal moves
    print(f"evaluator unavailable, playing random: {error}")
    _search = None

_board = chess.Board()  # our view of the game, kept in sync across calls


def _sync(fen: str) -> chess.Board:
    """Return our running board advanced to ``fen``, keeping history when we can."""
    global _board
    target = chess.Board(fen)
    if _matches(_board, target):
        return _board
    for move in _board.legal_moves:  # the opponent's reply gets us there
        _board.push(move)
        if _matches(_board, target):
            return _board
        _board.pop()
    _board = target  # desynced (first move, or a skipped position): history is lost
    return _board


def _matches(a: chess.Board, b: chess.Board) -> bool:
    return (
        a.board_fen() == b.board_fen()
        and a.turn == b.turn
        and a.castling_rights == b.castling_rights
        and a.ep_square == b.ep_square
    )


def _budget_ms(time_left_ms: int, fullmove: int) -> float:
    if time_left_ms < _PANIC_MS:
        return max(20.0, min(time_left_ms * 0.08, 400.0))
    moves_left = max(14, 44 - fullmove)
    target = time_left_ms / moves_left + 300.0  # spend the clock plus most of the increment
    return min(target, time_left_ms - _SAFETY_MS)


def get_move(fen: str, time_left_ms: int) -> str:
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1 or _search is None:
        move = legal[0] if len(legal) == 1 else random.choice(legal)
        board.push(move)
        return move.uci()

    try:
        deadline = time.monotonic() + _budget_ms(time_left_ms, board.fullmove_number) / 1000.0
        move = _search.run(board, deadline)
    except Exception as error:  # never forfeit on a bug
        print(f"search failed, playing a legal move: {error}")
        move = legal[0]

    board.push(move)
    return move.uci()
