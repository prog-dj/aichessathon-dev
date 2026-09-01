"""A cheap material + piece-square evaluation. Shipped in the zip.

The network's value head is accurate about *who* is winning but compresses *by
how much* - it says +0.4 where the position is +0.8. The search blends this
material term back in so a hung piece reads as a hung piece. Pure counting, a
few microseconds, never wrong about material.
"""

from __future__ import annotations

import math

import chess

_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# midgame piece-square tables, White's view, a1..h8; Black mirrors vertically.
# fmt: off
_PAWN = (
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
)
_KNIGHT = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
)
_BISHOP = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
)
_ROOK = (
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
)
_QUEEN = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
)
_KING_MID = (
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
)
_KING_END = (
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-25,  0,  0,  0,  0,-25,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50,
)
_PAWN_END = (
      0,  0,  0,  0,  0,  0,  0,  0,
     10, 10, 10, 10, 10, 10, 10, 10,
     15, 15, 15, 15, 15, 15, 15, 15,
     25, 25, 25, 25, 25, 25, 25, 25,
     45, 45, 45, 45, 45, 45, 45, 45,
     80, 80, 80, 80, 80, 80, 80, 80,
    130,130,130,130,130,130,130,130,
      0,  0,  0,  0,  0,  0,  0,  0,
)
# fmt: on
_MID = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK: _ROOK,
    chess.QUEEN: _QUEEN,
    chess.KING: _KING_MID,
}
_END = {**_MID, chess.PAWN: _PAWN_END, chess.KING: _KING_END}

_PHASE_MAX = 24  # 2*(N 1 + B 1 + R 2) + 2*(Q 4), both sides at the start

_BISHOP_PAIR = 30
_TEMPO = 12


def _phase(board: chess.Board) -> float:
    total = (
        chess.popcount(board.knights)
        + chess.popcount(board.bishops)
        + 2 * chess.popcount(board.rooks)
        + 4 * chess.popcount(board.queens)
    )
    return min(total, _PHASE_MAX) / _PHASE_MAX


_STATIC = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)  # tables don't change by phase
_PHASED = (chess.PAWN, chess.KING)


def material_pst_cp(board: chess.Board) -> int:
    """Material + piece-square score, centipawns, side-to-move point of view.

    The pawn and king tables interpolate between a midgame and an endgame set by
    game phase, so the king centralises once the queens come off.
    """
    white = 0.0
    for piece_type in _STATIC:
        table = _MID[piece_type]
        base = _VALUE[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            white += base + table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            white -= base + table[square ^ 56]

    phase = _phase(board)
    other = 1.0 - phase
    for piece_type in _PHASED:
        mid, end = _MID[piece_type], _END[piece_type]
        base = _VALUE.get(piece_type, 0)
        for square in board.pieces(piece_type, chess.WHITE):
            white += base + phase * mid[square] + other * end[square]
        for square in board.pieces(piece_type, chess.BLACK):
            flipped = square ^ 56
            white -= base + phase * mid[flipped] + other * end[flipped]

    return int(white) if board.turn == chess.WHITE else -int(white)


def evaluate_cp(board: chess.Board) -> int:
    """The search evaluation: material + piece-square, plus a couple of cheap terms."""
    stm = board.turn
    score = material_pst_cp(board) + _TEMPO

    white_pair = len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2
    black_pair = len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    pair = _BISHOP_PAIR * (white_pair - black_pair)
    score += pair if stm == chess.WHITE else -pair
    return score


def cp_to_scalar(cp: float, scale: float = 400.0) -> float:
    """Squash a centipawn score into [-1, 1] to match the value head."""
    return math.tanh(cp / scale)
