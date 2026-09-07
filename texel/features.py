"""Feature extraction for Texel tuning — an exact mirror of
``fastchess.evaluate_hce`` decomposed into (feature, weight) pairs.

`eval_white_cp(feat, W) == fastchess.evaluate_hce(fen)` (white POV, +/-1 cp of
integer-division rounding) when W == the current weights.  ``verify.py`` checks
this on a random sample before any tuning run.

The eval is a tapered sum:  score = (MG*phase + EG*(24-phase)) // 24  + tempo
so every feature is tagged mg / eg / both.  King danger is quadratic in its
sub-weights, so it is kept as raw per-piece ring-hit counts and the non-linear
curve is applied in the model, not here.
"""
from __future__ import annotations

import numpy as np
import chess

# ---- static tables mirrored from fastchess -------------------------------
_FILE_BB = [chess.BB_FILES[f] for f in range(8)]
_ADJ = []
for f in range(8):
    m = 0
    if f > 0:
        m |= _FILE_BB[f - 1]
    if f < 7:
        m |= _FILE_BB[f + 1]
    _ADJ.append(m)

_PC = chess.popcount


def _bishop_from(sq: int, occ: int) -> int:
    return chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]


def _rook_from(sq: int, occ: int) -> int:
    return (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
            | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])


def _passed_mask(sq: int, white: bool) -> int:
    """Squares in front of `sq` on its file and the two adjacent files."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    files = _FILE_BB[f] | _ADJ[f]
    if white:
        ahead = 0
        for rr in range(r + 1, 8):
            ahead |= chess.BB_RANKS[rr]
    else:
        ahead = 0
        for rr in range(0, r):
            ahead |= chess.BB_RANKS[rr]
    return files & ahead


# feature layout ----------------------------------------------------------
#  material  : 5 mg + 5 eg          (P N B R Q, white-minus-black counts)
#  mobility  : 4 mg                 (N B R Q, sum of (attacks & ~own) popcount, w-b)
#  king dngr : 8 raw counts         (w: N B R Q ring-hits on black king; b: on white king)
#  bishop pr : 1 mg + 1 eg          (w_has - b_has, in {-1,0,1})
#  isolated  : 1 mg + 1 eg          (w_count - b_count)
#  doubled   : 1 mg + 1 eg          (w_count - b_count)
#  passed    : 6 mg + 6 eg          (by rank-from-own-side 1..6, w-b)
#  rook open : 1 mg                 (w - b)
#  rook half : 1 mg                 (w - b)
#  pst       : 384 mg + 384 eg      (per (piece,sq) white-minus-black occupancy)
#  tempo     : 1 (post-taper, +1 if white to move else -1)
#  phase     : scalar 0..24 (not a weight — the taper interpolant)
N_MAT = 5
N_MOB = 4
N_KD = 8
N_KS = 4          # [weak_w, safe_w, weak_b, safe_b]  (_w = white attacking black king)
N_PST = 384


def extract(fen: str) -> dict:
    b = chess.Board(fen)
    occ = b.occupied
    wocc = b.occupied_co[chess.WHITE]
    bocc = b.occupied_co[chess.BLACK]

    # phase
    phase = 0
    for pt, w in ((chess.KNIGHT, 1), (chess.BISHOP, 1), (chess.ROOK, 2), (chess.QUEEN, 4)):
        phase += w * (chess.popcount(b.pieces_mask(pt, chess.WHITE))
                      + chess.popcount(b.pieces_mask(pt, chess.BLACK)))
    phase = min(phase, 24)

    mat = np.zeros(N_MAT, np.float64)
    mob = np.zeros(N_MOB, np.float64)
    kd = np.zeros(N_KD, np.float64)          # [wN wB wR wQ | bN bB bR bQ]
    pst_mg = np.zeros(N_PST, np.float64)
    pst_eg = np.zeros(N_PST, np.float64)

    wk = b.king(chess.WHITE)
    bk = b.king(chess.BLACK)
    w_ring = int(chess.BB_KING_ATTACKS[bk]) | (1 << bk)   # attacking BLACK king
    b_ring = int(chess.BB_KING_ATTACKS[wk]) | (1 << wk)   # attacking WHITE king

    # per-colour aggregate attack sets (pre own-occupancy mask) for weak-square
    # and safe-check king-safety features
    a_all = {chess.WHITE: 0, chess.BLACK: 0}
    a_n = {chess.WHITE: 0, chess.BLACK: 0}
    a_d = {chess.WHITE: 0, chess.BLACK: 0}   # bishop + queen
    a_o = {chess.WHITE: 0, chess.BLACK: 0}   # rook + queen

    for col in (chess.WHITE, chess.BLACK):
        sign = 1.0 if col == chess.WHITE else -1.0
        own = wocc if col == chess.WHITE else bocc
        ring = w_ring if col == chess.WHITE else b_ring
        kd_base = 0 if col == chess.WHITE else 4
        x = b.pieces_mask(chess.PAWN, col)
        while x:
            sq = (x & -x).bit_length() - 1
            x &= x - 1
            a_all[col] |= int(chess.BB_PAWN_ATTACKS[col][sq])
        for pt in range(1, 7):               # PAWN..KING (chess uses 1..6)
            pcs = b.pieces_mask(pt, col)
            x = pcs
            while x:
                sq = (x & -x).bit_length() - 1
                x &= x - 1
                idx = sq if col == chess.WHITE else (sq ^ 56)
                p0 = pt - 1                   # 0..5
                mat_i = p0 if p0 < 5 else -1
                if mat_i >= 0:
                    mat[mat_i] += sign
                pst_mg[p0 * 64 + idx] += sign
                pst_eg[p0 * 64 + idx] += sign
                if 1 <= p0 <= 4:              # N B R Q
                    raw = int(b.attacks_mask(sq))
                    a_all[col] |= raw
                    if p0 == 1:
                        a_n[col] |= raw
                    elif p0 == 2:
                        a_d[col] |= raw
                    elif p0 == 3:
                        a_o[col] |= raw
                    else:
                        a_d[col] |= raw
                        a_o[col] |= raw
                    att = raw & ~own
                    pc = chess.popcount(att)
                    mob[p0 - 1] += sign * pc
                    rh = chess.popcount(att & ring)
                    kd[kd_base + (p0 - 1)] += rh

    # weak squares (king-ring, attacked by enemy, undefended) + safe checks
    # (checking square an enemy piece reaches that we don't defend). ks =
    # [weak_w, safe_w, weak_b, safe_b] where _w is WHITE attacking the black king.
    ks = np.zeros(4, np.float64)
    for col in (chess.WHITE, chess.BLACK):
        them = not col
        ek = bk if col == chess.WHITE else wk
        own_c = wocc if col == chess.WHITE else bocc
        ac, at = a_all[col], a_all[them]
        ering = int(chess.BB_KING_ATTACKS[ek]) | (1 << ek)
        weak = _PC(ering & ac & ~at)
        nchk = _PC(int(chess.BB_KNIGHT_ATTACKS[ek]) & a_n[col] & ~own_c & ~at)
        bchk = _PC(_bishop_from(ek, occ) & a_d[col] & ~own_c & ~at)
        rchk = _PC(_rook_from(ek, occ) & a_o[col] & ~own_c & ~at)
        base = 0 if col == chess.WHITE else 2
        ks[base] = weak
        ks[base + 1] = nchk + bchk + rchk

    # bishop pair
    bp = (1.0 if chess.popcount(b.pieces_mask(chess.BISHOP, chess.WHITE)) >= 2 else 0.0) \
        - (1.0 if chess.popcount(b.pieces_mask(chess.BISHOP, chess.BLACK)) >= 2 else 0.0)

    iso = 0.0
    dbl = 0.0
    passed_mg = np.zeros(6, np.float64)
    rook_open = 0.0
    rook_half = 0.0
    for col in (chess.WHITE, chess.BLACK):
        sign = 1.0 if col == chess.WHITE else -1.0
        own_p = b.pieces_mask(chess.PAWN, col)
        enemy_p = b.pieces_mask(chess.PAWN, not col)
        x = own_p
        while x:
            sq = (x & -x).bit_length() - 1
            x &= x - 1
            f = chess.square_file(sq)
            if not (own_p & _ADJ[f]):
                iso += sign
            if chess.popcount(own_p & _FILE_BB[f]) > 1:
                dbl += sign
            if not (_passed_mask(sq, col == chess.WHITE) & enemy_p):
                rr = chess.square_rank(sq) if col == chess.WHITE else 7 - chess.square_rank(sq)
                if 1 <= rr <= 6:
                    passed_mg[rr - 1] += sign
        x = b.pieces_mask(chess.ROOK, col)
        while x:
            sq = (x & -x).bit_length() - 1
            x &= x - 1
            fm = _FILE_BB[chess.square_file(sq)]
            if not (fm & b.pieces_mask(chess.PAWN, chess.WHITE) | fm & b.pieces_mask(chess.PAWN, chess.BLACK)):
                rook_open += sign
            elif not (fm & own_p):
                rook_half += sign

    return dict(
        phase=float(phase),
        wtm=1.0 if b.turn == chess.WHITE else -1.0,
        mat=mat, mob=mob, kd=kd, ks=ks,
        bp=bp, iso=iso, dbl=dbl,
        passed=passed_mg,
        rook_open=rook_open, rook_half=rook_half,
        pst_mg=pst_mg, pst_eg=pst_eg,
    )
