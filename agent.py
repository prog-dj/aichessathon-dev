"""The submission entrypoint. The platform imports this file and calls get_move.

The move is chosen by ``fastchess`` - a numba-JIT bitboard engine (magic
move generation, iterative-deepening PVS alpha-beta with a transposition
table, killers, history, null move, LMR and a tapered hand evaluation),
compiled to native code at import. If numba is missing or a compile fails
we fall back to the pure-Python engine in ``alphabeta.py``.

This file owns the two things the engine needs from outside the position:
the game's move history (so repetitions are seen) and a per-move time budget.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import chess

_WEIGHTS = Path(__file__).with_name("weights")
_SAFETY_MS = 500  # never plan to use the last half second of the clock
_PANIC_MS = 4_000
_INCREMENT_MS = 500  # fixed 0.5s/move from the tournament time control

# --- the numba engine, compiled at import --------------------------------
_fast = None
try:
    import fastchess

    _fast = fastchess.Engine()  # type: ignore[no-untyped-call]
    print("engine: fastchess (numba)")
except Exception as error:
    print(f"fastchess unavailable, using the python fallback: {error}")

# --- pure-Python fallback ------------------------------------------------
_fallback = None
if _fast is None:
    try:
        from alphabeta import AlphaBetaSearch
        from inference import Evaluator

        try:
            _fallback = AlphaBetaSearch(Evaluator(_WEIGHTS / "model.onnx"))
        except Exception as err:
            print(f"fallback evaluator unavailable: {err}")
            _fallback = AlphaBetaSearch(None)
    except Exception as err:
        print(f"python fallback unavailable: {err}")

_book = None
try:
    from book import Book

    _book = Book(_WEIGHTS / "book.json")
    print(f"opening book: {len(_book)} positions")
except Exception as error:
    print(f"no opening book: {error}")

_tablebase = None
try:
    from endgame import Tablebase

    _tablebase = Tablebase(_WEIGHTS / "syzygy")
    print("syzygy tablebase loaded")
except Exception as error:
    print(f"no tablebase: {error}")

_board = chess.Board()
_root_fen = chess.STARTING_FEN


def _sync(fen: str) -> chess.Board:
    """Return our running board advanced to ``fen``, keeping history when we can."""
    global _board, _root_fen
    target = chess.Board(fen)
    if _matches(_board, target):
        return _board
    for move in _board.legal_moves:
        _board.push(move)
        if _matches(_board, target):
            return _board
        _board.pop()
    _board = target
    _root_fen = fen
    return _board


def _matches(a: chess.Board, b: chess.Board) -> bool:
    return (
        a.board_fen() == b.board_fen()
        and a.turn == b.turn
        and a.castling_rights == b.castling_rights
        and a.ep_square == b.ep_square
    )


def _budget_ms(time_left_ms: int, fullmove: int) -> tuple[float, float]:
    """(soft target, hard cap) in milliseconds."""
    if time_left_ms < _PANIC_MS:
        t = max(20.0, min(time_left_ms * 0.08, 400.0))
        return t, t
    moves_left = max(20, 55 - fullmove)
    soft = (time_left_ms - _SAFETY_MS) / moves_left + 0.9 * _INCREMENT_MS
    soft = min(soft, time_left_ms - _SAFETY_MS)
    hard = min(soft * 1.8, time_left_ms - _SAFETY_MS)
    return soft, hard


def get_move(fen: str, time_left_ms: int) -> str:
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        board.push(legal[0])
        return legal[0].uci()

    chosen: chess.Move | None = None

    if _book is not None and board.fullmove_number <= 16:
        try:
            chosen = _book.move(board)
        except Exception as error:
            print(f"book lookup failed: {error}")

    if chosen is None:
        soft, hard = _budget_ms(time_left_ms, board.fullmove_number)
        chosen = _fast_move(soft, hard) or _fallback_move(board, soft)
        if chosen is None:
            chosen = legal[0]

    chosen = _tablebase_guard(board, chosen)
    board.push(chosen)
    return chosen.uci()


def _fast_move(soft: float, hard: float) -> chess.Move | None:
    if _fast is None:
        return None
    try:
        history = [m.uci() for m in _board.move_stack]
        uci, _score, _depth, _nodes = _fast.best_move(  # type: ignore[no-untyped-call]
            _root_fen, history, int(soft), int(hard)
        )
        return chess.Move.from_uci(uci)
    except Exception as error:  # never forfeit on an engine bug
        print(f"engine move failed, falling back: {error}")
        return None


def _fallback_move(board: chess.Board, soft: float) -> chess.Move | None:
    if _fallback is None:
        return random.choice(list(board.legal_moves))
    try:
        return _fallback.run(board, time.monotonic() + soft / 1000.0)
    except Exception as error:
        print(f"fallback search failed: {error}")
        return None


def _tablebase_guard(board: chess.Board, chosen: chess.Move) -> chess.Move:
    """Keep the chosen move only if it preserves the tablebase result."""
    if _tablebase is None:
        return chosen
    try:
        approved = _tablebase.best_moves(board)
    except Exception as error:
        print(f"tablebase probe failed: {error}")
        return chosen
    if approved is None or chosen in approved:
        return chosen
    return approved[0]
