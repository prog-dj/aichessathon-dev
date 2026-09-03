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
from book import Book
from endgame import Tablebase
from inference import Evaluator

_WEIGHTS = Path(__file__).with_name("weights")
_SAFETY_MS = 500  # never plan to use the last half second of the clock
_PANIC_MS = 4_000
_INCREMENT_MS = 500  # fixed 0.5s/move from the tournament time control

try:
    _search: AlphaBetaSearch | None = AlphaBetaSearch(Evaluator(_WEIGHTS / "model.onnx"))
except Exception as error:  # missing or broken weights: search on material alone
    print(f"evaluator unavailable, searching without the net: {error}")
    try:
        _search = AlphaBetaSearch(None)
    except Exception:
        _search = None

_book: Book | None
try:
    _book = Book(_WEIGHTS / "book.json")
    print(f"opening book: {len(_book)} positions")
except Exception as error:  # no book, or unreadable: just search from move one
    print(f"no opening book: {error}")
    _book = None

_tablebase: Tablebase | None
try:
    _tablebase = Tablebase(_WEIGHTS / "syzygy")
    print("syzygy tablebase loaded")
except Exception as error:  # no tables: the search handles endgames on its own
    print(f"no tablebase: {error}")
    _tablebase = None

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
    # base share of the remaining clock, plus almost the whole increment - the
    # increment refills every move, so spending it keeps the base clock flat.
    target = (time_left_ms - _SAFETY_MS) / moves_left + 0.9 * _INCREMENT_MS
    return min(target, time_left_ms - _SAFETY_MS)


def get_move(fen: str, time_left_ms: int) -> str:
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"

    chosen: chess.Move | None = None
    if len(legal) == 1 or _search is None:
        chosen = legal[0] if len(legal) == 1 else random.choice(legal)
        board.push(chosen)
        return chosen.uci()

    if _book is not None and board.fullmove_number <= 16:
        try:
            chosen = _book.move(board)
        except Exception as error:
            print(f"book lookup failed: {error}")

    if chosen is None:
        try:
            deadline = time.monotonic() + _budget_ms(time_left_ms, board.fullmove_number) / 1000.0
            chosen = _search.run(board, deadline)
        except Exception as error:  # never forfeit on a bug
            print(f"search failed, playing a legal move: {error}")
            chosen = legal[0]

    chosen = _tablebase_guard(board, chosen)
    board.push(chosen)
    return chosen.uci()


def _tablebase_guard(board: chess.Board, chosen: chess.Move) -> chess.Move:
    """Keep the search's move only if it preserves the tablebase result."""
    if _tablebase is None:
        return chosen
    try:
        approved = _tablebase.best_moves(board)
    except Exception as error:
        print(f"tablebase probe failed: {error}")
        return chosen
    if approved is None or chosen in approved:
        return chosen
    scores = _search._root_scores if _search is not None else {}
    return max(approved, key=lambda m: scores.get(m, 0.0))
