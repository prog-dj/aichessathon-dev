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

_EXPECTED_GAME_LENGTH = 90  # real games run this long (round 17: 88 moves) - the old
# formula's implicit horizon of ~55 meant moves_left floored at 20 forever past move 35,
# regardless of how many moves were actually left, which is what let the clock run down
# to 3-4s for the entire second half of that game.
_MIN_MOVES_LEFT = 20
_PACE_RATIO_MIN = 0.4   # behind pace -> shrink the budget hard
_PACE_RATIO_MAX = 1.15  # ahead of pace -> only a mild bump, deliberately asymmetric since
# the evidenced failure is 100% overspending, and letting the engine spend *more* on a
# signal is exactly the shape of an earlier, reverted regression - lean toward banking time.

_pace_ref_time_left_ms: float | None = None
_pace_ref_fullmove: int | None = None

# Complexity weighting: the pace fix above stops the clock from crashing, but it
# spreads the *preserved* time roughly evenly across whatever moves are left,
# including ones that turn out to be near-forced simplified endgames (a king
# marching in to support a passed pawn needs little real calculation). Real
# mistakes happen when there's still material on the board and genuine choices
# to weigh, not when the position has already boiled down to something close to
# mechanical - so scale the budget by how much non-pawn material remains, on
# top of (not instead of) the pace correction.
_COMPLEXITY_PIECE_VALUES = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3}
_MAX_COMPLEXITY_UNITS = 2 * sum(_COMPLEXITY_PIECE_VALUES.values())  # both sides at full strength
_COMPLEXITY_MULT_MIN = 0.6  # bare-material / near-forced endgame: spend less
_COMPLEXITY_MULT_MAX = 1.3  # full material still on: spend more


def _complexity_multiplier(board: chess.Board) -> float:
    units = sum(
        value * (len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK)))
        for piece_type, value in _COMPLEXITY_PIECE_VALUES.items()
    )
    ratio = min(1.0, units / _MAX_COMPLEXITY_UNITS)
    return _COMPLEXITY_MULT_MIN + (_COMPLEXITY_MULT_MAX - _COMPLEXITY_MULT_MIN) * ratio

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


def _budget_ms(time_left_ms: int, fullmove: int, complexity_mult: float = 1.0) -> tuple[float, float]:
    """(soft target, hard cap) in milliseconds.

    Tracks whether we're ahead of or behind an idealised clock-usage curve for the
    whole game (not just this move) and tightens or loosens the budget accordingly.
    This is a cumulative, whole-game correction on the *budget calculation* - it does
    not look at search/PV behaviour at all, unlike an earlier, reverted attempt that
    varied a single move's own search cutoff based on that move's PV stability.

    `complexity_mult` (see _complexity_multiplier) then scales the pace-corrected
    budget up or down for this specific position - not applied in panic mode,
    which stays a pure last-resort emergency behaviour regardless of complexity.
    """
    global _pace_ref_time_left_ms, _pace_ref_fullmove

    if time_left_ms < _PANIC_MS:
        t = max(20.0, min(time_left_ms * 0.08, 400.0))
        return t, t

    if _pace_ref_time_left_ms is None:
        _pace_ref_time_left_ms = float(time_left_ms)
        _pace_ref_fullmove = fullmove

    moves_since_ref = max(1, fullmove - _pace_ref_fullmove)
    expected_moves_left_at_ref = max(_MIN_MOVES_LEFT, _EXPECTED_GAME_LENGTH - _pace_ref_fullmove)
    ideal_per_move = (_pace_ref_time_left_ms - _SAFETY_MS) / expected_moves_left_at_ref
    expected_time_left_now = (
        _pace_ref_time_left_ms
        - ideal_per_move * moves_since_ref
        + _INCREMENT_MS * moves_since_ref
    )

    pace_ratio = time_left_ms / max(1.0, expected_time_left_now)
    pace_ratio = min(max(pace_ratio, _PACE_RATIO_MIN), _PACE_RATIO_MAX)

    moves_left = max(_MIN_MOVES_LEFT, _EXPECTED_GAME_LENGTH - fullmove)
    moves_left = moves_left / pace_ratio

    soft = (time_left_ms - _SAFETY_MS) / moves_left + 0.9 * _INCREMENT_MS
    soft *= complexity_mult
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
        soft, hard = _budget_ms(
            time_left_ms, board.fullmove_number, _complexity_multiplier(board)
        )
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
