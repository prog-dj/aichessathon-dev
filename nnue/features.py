"""HalfKA-style feature mapping, shared by prepare.py, train_t4.py and the
numba inference in fastchess.py.  KEEP ALL THREE IN SYNC.

Index for a piece of type `pt` (0..5 = P..K), from a given perspective, at
perspective-relative square `sq_rel` (black's board pre-flipped by ^56), given
that perspective's own king square `king_sq_persp` (also pre-flipped):

    own  = (piece_colour == perspective)
    flat = (pt*2 + (0 if own else 1)) * 64 + sq_rel          # 0..767
    idx  = king_bucket(king_sq_persp) * 768 + flat

32 king buckets: (rank // 2) * 8 + file.  Rank-band + file — enough to learn
"king on the back rank vs. advanced" (most of king safety) without the full
64-square table that blows the FT cache on king-move rebuilds.
"""
from __future__ import annotations

import chess

N_KING_BUCKETS = 32
N_FEATURES = N_KING_BUCKETS * 768        # 24576
PAD = N_FEATURES                          # embedding padding_idx
MAX_PIECES = 32


def king_bucket(king_sq_persp: int) -> int:
    return ((king_sq_persp >> 3) >> 1) * 8 + (king_sq_persp & 7)


def feat_index(pt: int, own: bool, sq_rel: int, king_sq_persp: int) -> int:
    flat = (pt * 2 + (0 if own else 1)) * 64 + sq_rel
    return king_bucket(king_sq_persp) * 768 + flat


def board_features(board: chess.Board) -> tuple[list[int], list[int]]:
    """(white-perspective indices, black-perspective indices) for every piece."""
    kw = board.king(chess.WHITE)
    kb_rel = board.king(chess.BLACK) ^ 56
    iw: list[int] = []
    ib: list[int] = []
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type - 1
        white_piece = piece.color == chess.WHITE
        iw.append(feat_index(pt, white_piece, sq, kw))
        ib.append(feat_index(pt, not white_piece, sq ^ 56, kb_rel))
    return iw, ib
