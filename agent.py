"""The submission entrypoint. The platform imports this file and calls get_move.

An iterative-deepening alpha-beta search (alphabeta.py) over a material +
piece-square evaluation, with the network deciding among the root moves the
search rates as equal. This file owns the two things the search needs from
outside the position: a running board carrying the game's move history (so
repetitions are seen), and a time budget for the current move.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import chess

from alphabeta import AlphaBetaSearch
from inference import Evaluator

_MODEL_PATH = Path(__file__).with_name("weights") / "model.onnx"
_SAFETY_MS = 500  # never plan to use the last half second of the clock
_PANIC_MS = 4_000

try:
    _search: AlphaBetaSearch | None = AlphaBetaSearch(Evaluator(_MODEL_PATH))
except Exception as error:  # missing or broken weights: search on material alone
    print(f"evaluator unavailable, searching without the net: {error}")
    try:
        _search = AlphaBetaSearch(None)
    except Exception:
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
    moves_left = max(20, 55 - fullmove)
    target = time_left_ms / moves_left + 300.0  # the clock share plus part of the increment
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
