"""A cheap material + piece-square evaluation. Shipped in the zip.

The network's value head is accurate about *who* is winning but compresses *by
how much* - it says +0.4 where the position is +0.8. The search blends this
material term back in so a hung piece reads as a hung piece. Pure counting, a
few microseconds, never wrong about material.
"""

from __future__ import annotations

import math

import chess

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 338,
    chess.BISHOP: 350,
    chess.ROOK: 530,
    chess.QUEEN: 960,
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


_BASE = tuple(PIECE_VALUE.get(pt, 0) for pt in range(7))  # indexed by piece_type 1..6
_PHASED = frozenset((chess.PAWN, chess.KING))

# --- precomputed masks for the positional terms ---
_FILE_OF = tuple(chess.BB_FILES[chess.square_file(sq)] for sq in range(64))
_ADJ_FILES = tuple(
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in (chess.square_file(sq) for sq in range(64))
)


def _front_span(square: int, white: bool) -> int:
    rank = chess.square_rank(square)
    ranks = range(rank + 1, 8) if white else range(rank - 1, -1, -1)
    return sum(chess.BB_RANKS[r] for r in ranks)


_PASSED_W = tuple(
    (_FILE_OF[sq] | _ADJ_FILES[sq]) & _front_span(sq, True) for sq in range(64)
)
_PASSED_B = tuple(
    (_FILE_OF[sq] | _ADJ_FILES[sq]) & _front_span(sq, False) for sq in range(64)
)
_PASSED_BONUS = (0, 8, 13, 23, 42, 70, 106, 0)  # by rank of the pawn (0..7)
_ROOK_OPEN = 26
_ROOK_HALF_OPEN = 12
_KING_SHIELD = 16  # per pawn in front of a castled king


_MATE_THRESHOLD = 450  # once this far ahead, start driving the bare king to a corner


def _material_pst_white(board: chess.Board) -> float:
    phase = _phase(board)
    other = 1.0 - phase
    white = 0.0
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        sq = square if piece.color == chess.WHITE else square ^ 56
        if piece_type in _PHASED:
            table_value = phase * _MID[piece_type][sq] + other * _END[piece_type][sq]
        else:
            table_value = _MID[piece_type][sq]
        value = _BASE[piece_type] + table_value
        white += value if piece.color == chess.WHITE else -value
    return white


def material_pst_white(board: chess.Board) -> float:
    """Material + piece-square score from White's point of view (the incremental base)."""
    return _material_pst_white(board)


def material_pst_cp(board: chess.Board) -> int:
    """Material + piece-square score, centipawns, side-to-move point of view.

    The pawn and king tables interpolate between a midgame and an endgame set by
    game phase, so the king centralises once the queens come off.
    """
    white = _material_pst_white(board)
    return int(white) if board.turn == chess.WHITE else int(-white)


def evaluate_cp(board: chess.Board, material_white: float | None = None) -> int:
    """The search evaluation: material + piece-square + a handful of cheap terms.

    Assembled from White's point of view, then flipped once for the side to move.
    Pass ``material_white`` to reuse an incrementally-maintained material+PST base.
    """
    white = (
        _material_pst_white(board) if material_white is None else material_white
    ) + _positional(board)
    if abs(white) >= _MATE_THRESHOLD:
        white += _mate_drive_white(board, white > 0)
    if board.turn == chess.WHITE:
        return int(white) + _TEMPO
    return int(-white) + _TEMPO


def mpst_quiet_delta(board: chess.Board, move: chess.Move) -> float | None:
    """White-POV change in material_pst_white for a quiet, non-promotion move.

    Returns None for captures/promotions/castling/en passant - the caller then
    recomputes from scratch (phase can shift, rook also moves, etc.).
    """
    if move.promotion or board.is_capture(move) or board.is_castling(move):
        return None
    piece = board.piece_type_at(move.from_square)
    if piece is None:
        return None
    white = piece != 0 and board.color_at(move.from_square) == chess.WHITE
    src = move.from_square if white else move.from_square ^ 56
    dst = move.to_square if white else move.to_square ^ 56
    if piece in _PHASED:
        phase = _phase(board)
        other = 1.0 - phase
        mid, end = _MID[piece], _END[piece]
        delta = phase * (mid[dst] - mid[src]) + other * (end[dst] - end[src])
    else:
        table = _MID[piece]
        delta = float(table[dst] - table[src])
    return delta if white else -delta


def _positional(board: chess.Board) -> int:
    """Passed pawns, rooks on open files, king pawn shield - all White's point of view."""
    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    score = 0

    white_pair = chess.popcount(board.bishops & board.occupied_co[chess.WHITE]) >= 2
    black_pair = chess.popcount(board.bishops & board.occupied_co[chess.BLACK]) >= 2
    score += _BISHOP_PAIR * (white_pair - black_pair)

    for sq in chess.scan_forward(white_pawns):
        if not _PASSED_W[sq] & black_pawns:
            score += _PASSED_BONUS[chess.square_rank(sq)]
    for sq in chess.scan_forward(black_pawns):
        if not _PASSED_B[sq] & white_pawns:
            score -= _PASSED_BONUS[7 - chess.square_rank(sq)]

    for sq in chess.scan_forward(board.rooks & board.occupied_co[chess.WHITE]):
        file_bb = _FILE_OF[sq]
        if not file_bb & board.pawns:
            score += _ROOK_OPEN
        elif not file_bb & white_pawns:
            score += _ROOK_HALF_OPEN
    for sq in chess.scan_forward(board.rooks & board.occupied_co[chess.BLACK]):
        file_bb = _FILE_OF[sq]
        if not file_bb & board.pawns:
            score -= _ROOK_OPEN
        elif not file_bb & black_pawns:
            score -= _ROOK_HALF_OPEN

    score += _king_shield(board.king(chess.WHITE), white_pawns, True)
    score -= _king_shield(board.king(chess.BLACK), black_pawns, False)
    return score


def _king_shield(king: int | None, own_pawns: int, white: bool) -> int:
    if king is None:
        return 0
    rank = chess.square_rank(king)
    if (white and rank > 1) or (not white and rank < 6):
        return 0  # king has left home - shield term does not apply
    shield_rank = chess.BB_RANKS[rank + 1] if white else chess.BB_RANKS[rank - 1]
    files = _FILE_OF[king] | _ADJ_FILES[king]
    return _KING_SHIELD * chess.popcount(own_pawns & files & shield_rank) - _KING_SHIELD


def _mate_drive_white(board: chess.Board, white_winning: bool) -> int:
    """Push the losing king to the edge and bring the winning king up. White POV."""
    strong = chess.WHITE if white_winning else chess.BLACK
    strong_king, weak_king = board.king(strong), board.king(not strong)
    if strong_king is None or weak_king is None:
        return 0
    wf, wr = weak_king & 7, weak_king >> 3
    sf, sr = strong_king & 7, strong_king >> 3
    edge = max(abs(2 * wf - 7), abs(2 * wr - 7))  # 1 (centre) .. 7 (corner)
    closeness = 14 - (abs(wf - sf) + abs(wr - sr))
    bonus = 6 * edge + 3 * closeness
    return bonus if white_winning else -bonus


def cp_to_scalar(cp: float, scale: float = 400.0) -> float:
    """Squash a centipawn score into [-1, 1] to match the value head."""
    return math.tanh(cp / scale)
