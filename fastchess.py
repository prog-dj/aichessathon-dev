"""A numba-JIT bitboard chess engine: magic-bitboard move generation, an
iterative-deepening alpha-beta search and a tapered hand evaluation.

Everything hot is compiled to native code by numba at import. The board is a
small ``uint64`` array plus an ``int8`` mailbox, passed explicitly; the search
keeps its tables in module globals (one process, one thread).

``agent.py`` uses :class:`Engine`; it falls back to ``alphabeta.py`` if numba
is missing or a JIT compile fails.
"""

from __future__ import annotations

import numpy as np
from numba import njit

U = np.uint64
I = np.int64

# --- board layout ----------------------------------------------------------
# bb[0..5]  piece bitboards: pawn knight bishop rook queen king
# bb[6]     white occupancy      bb[7] black occupancy      bb[8] all occupancy
# bb[9]     side to move (0 white, 1 black)
# bb[10]    castling rights mask (WK1 WQ2 BK4 BQ8)
# bb[11]    en-passant target square, 64 if none
# bb[12]    halfmove clock        bb[13] zobrist key        bb[14] fullmove
BB_LEN = 16
WOCC, BOCC, OCC, STM, CAST, EP, HALF, KEY, FULL = 6, 7, 8, 9, 10, 11, 12, 13, 14

WK, WQ, BK, BQ = 1, 2, 4, 8

MAX_PLY = 128

ONE = U(1)
FULL_BB = U(0xFFFFFFFFFFFFFFFF)
_S = [U(i) for i in range(64)]

FILE_A = U(0x0101010101010101)
FILE_H = U(0x8080808080808080)
RANK_1 = U(0x00000000000000FF)
RANK_8 = U(0xFF00000000000000)
NOT_A = FILE_A ^ FULL_BB
NOT_H = FILE_H ^ FULL_BB

M1 = U(0x5555555555555555)
M2 = U(0x3333333333333333)
M4 = U(0x0F0F0F0F0F0F0F0F)
H01 = U(0x0101010101010101)
DEBRUIJN = U(0x03F79D71B4CB0A89)

_IDX64 = np.array(
    [0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53,
     51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54,
     35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13,
     8, 7, 6],
    dtype=np.int64,
)


@njit(cache=False, inline="always")
def popcount(x):
    x = U(x)
    x = x - ((x >> U(1)) & M1)
    x = (x & M2) + ((x >> U(2)) & M2)
    x = (x + (x >> U(4))) & M4
    return I((x * H01) >> U(56))


@njit(cache=False, inline="always")
def lsb(x):
    x = U(x)
    return _IDX64[I(((x & (U(0) - x)) * DEBRUIJN) >> U(58))]


# --- attack tables, generated once in Python at import --------------------


def _slide(sq, occ, dirs):
    attacks = 0
    f0, r0 = sq & 7, sq >> 3
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = r * 8 + f
            attacks |= 1 << s
            if occ & (1 << s):
                break
            f += df
            r += dr
    return attacks


_BDIR = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_RDIR = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _build_leapers():
    knight = np.zeros(64, np.uint64)
    king = np.zeros(64, np.uint64)
    pawn = np.zeros((2, 64), np.uint64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                knight[sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    king[sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for df in (-1, 1):
            if 0 <= f + df < 8:
                if r + 1 < 8:
                    pawn[0, sq] |= np.uint64(1) << np.uint64((r + 1) * 8 + f + df)
                if r - 1 >= 0:
                    pawn[1, sq] |= np.uint64(1) << np.uint64((r - 1) * 8 + f + df)
    return knight, king, pawn


def _edges(sq):
    f, r = sq & 7, sq >> 3
    m = 0
    for s in range(64):
        sf, sr = s & 7, s >> 3
        if (sr in (0, 7) and sr != r) or (sf in (0, 7) and sf != f):
            m |= 1 << s
    return m


# Fixed magic multipliers (found offline with the usual sparse-random search).
# Hardcoded so startup is a table fill, not a ~25 s search.
_B_MAGIC_HEX = (
    0x24400447848A0680, 0x20280AC404420020, 0x1008120152110A10, 0x0884140082000000,
    0x00011040A0430840, 0x1408411820022024, 0x0442012416C00800, 0x0108210048201800,
    0x0000680825080200, 0x0400810088A00C3, 0x0005080281020400, 0x0233040402908040,
    0x0904040420000002, 0x2000011022104205, 0x0404051802122000, 0x00002280C4022008,
    0x4050042810100092, 0x011040A04C0080A0, 0x008C0042080600A8, 0x01840C0124008010,
    0x1002010402940020, 0x510A00C041101D00, 0x00440000840C0A40, 0x0025004045029000,
    0x20F8410CA0020280, 0x0115040020888203, 0x0008084124024204, 0x2000404044010200,
    0x0118840000802020, 0x0202102002009000, 0x0009206002021000, 0x2241010002005100,
    0x00410C014440401, 0x0242022030904100, 0x2012180800040045, 0x2000020080480080,
    0x3030020020020082, 0x4204100090020800, 0x1050048304808404, 0x008104108428220C,
    0x4000C40420484100, 0x2004020210400205, 0x0C100C0144008800, 0x4648004202230802,
    0x0083080304014044, 0x0040106041402080, 0x20B04401004C042A, 0x1048810400801C20,
    0x0020841008041005, 0x0082444444200081, 0x0008390407040000, 0x0240090084240004,
    0x1000C004904C0243, 0x0010200404882420, 0x0021185040818000, 0x0121080081808511,
    0x0101010800821845, 0x0108212596301000, 0x0801040042009040, 0x0000888103094808,
    0x001404180490C401, 0x10001420A0028280, 0x4680102002240040, 0x3004010C11041100,
)
_R_MAGIC_HEX = (
    0x0280021184254000, 0x7040042000403000, 0x0080200080081000, 0x008028001001801C,
    0x0080080180020400, 0x0100280284000100, 0x0B800F0006003880, 0x2000100C2112084,
    0x0880008120C000, 0x0202404000201000, 0x0080801000200080, 0x2002001142031820,
    0x2040808004000800, 0x0042001882001005, 0x3081000200040500, 0x0200801140801100,
    0x0040008000402088, 0x1148210040008100, 0x0000820020420490, 0x0001030008100460,
    0x0108450011000800, 0x2000808022010400, 0x0020440001502608, 0x0000020000409124,
    0x2144408200210200, 0x0080400100208900, 0x0402048200204012, 0x2000890100221000,
    0x1102000A00102004, 0x102A4008011024A0, 0x0681000300120004, 0x0080460000A401,
    0x0400C00028800480, 0x0000201000400040, 0x1080847000802000, 0x0916860800801000,
    0x0800804400800800, 0x4002004802000410, 0x0004081084001201, 0x0A002080CA000104,
    0x0081400080628000, 0x041000C020004000, 0x11008122004A0010, 0x0040080010008080,
    0x00200801000D0010, 0x0082040002008080, 0x540100220003000C, 0x00400401A04A0001,
    0x04800A20490100, 0x50C0400820008480, 0x0420401020030100, 0x4300082D0080080,
    0x2308010008500500, 0x0140440002008080, 0x0211008402005100, 0x0042802300004080,
    0x0014410020308001, 0x0101022280104009, 0x0202003208814022, 0x0002050021300009,
    0x008200200428900E, 0x00A1000812240021, 0x0114050208009004, 0x4000290400822042,
)


def _build_magic(dirs, magic_hex):
    magics = np.array(magic_hex, np.uint64)
    masks = np.zeros(64, np.uint64)
    shifts = np.zeros(64, np.uint64)
    offsets = np.zeros(64, np.int64)
    table_parts = []
    total = 0
    for sq in range(64):
        mask = _slide(sq, 0, dirs) & ~_edges(sq)
        masks[sq] = mask
        bits = bin(mask).count("1")
        shifts[sq] = 64 - bits
        size = 1 << bits
        slot = np.zeros(size, np.uint64)
        sub = 0
        while True:
            idx = ((sub * int(magic_hex[sq])) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)
            slot[idx] = np.uint64(_slide(sq, sub, dirs))
            sub = (sub - mask) & mask
            if sub == 0:
                break
        offsets[sq] = total
        total += size
        table_parts.append(slot)
    return magics, masks, shifts, offsets, np.concatenate(table_parts)


KNIGHT_ATT, KING_ATT, PAWN_ATT = _build_leapers()
B_MAGIC, B_MASK, B_SHIFT, B_OFF, B_TAB = _build_magic(_BDIR, _B_MAGIC_HEX)
R_MAGIC, R_MASK, R_SHIFT, R_OFF, R_TAB = _build_magic(_RDIR, _R_MAGIC_HEX)


def _build_between_line():
    between = np.zeros((64, 64), np.uint64)
    line = np.zeros((64, 64), np.uint64)
    for a in range(64):
        for b in range(64):
            if a == b:
                continue
            for dirs in (_BDIR, _RDIR):
                aa = _slide(a, 0, dirs)
                if aa & (1 << b):
                    between[a, b] = np.uint64(_slide(a, 1 << b, dirs) & _slide(b, 1 << a, dirs))
                    line[a, b] = np.uint64(
                        (aa & _slide(b, 0, dirs)) | (1 << a) | (1 << b)
                    )
    return between, line


BETWEEN, LINE = _build_between_line()


@njit(cache=False, inline="always")
def bishop_attacks(sq, occ):
    o = U(occ) & B_MASK[sq]
    idx = (o * B_MAGIC[sq]) >> B_SHIFT[sq]
    return B_TAB[B_OFF[sq] + I(idx)]


@njit(cache=False, inline="always")
def rook_attacks(sq, occ):
    o = U(occ) & R_MASK[sq]
    idx = (o * R_MAGIC[sq]) >> R_SHIFT[sq]
    return R_TAB[R_OFF[sq] + I(idx)]


@njit(cache=False, inline="always")
def queen_attacks(sq, occ):
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


# --- move encoding -------------------------------------------------------------
# int32: bits 0..6 from, 6..12 to, 12..16 flag
QUIET, DPUSH, CK, CQ, CAP, EP_FLAG = 0, 1, 2, 3, 4, 5
PN, PB, PR, PQ = 8, 9, 10, 11
PNC, PBC, PRC, PQC = 12, 13, 14, 15


@njit(cache=False, inline="always")
def mk_move(frm, to, flag):
    return np.int32(frm | (to << 6) | (flag << 12))


@njit(cache=False, inline="always")
def m_from(m):
    return m & 63


@njit(cache=False, inline="always")
def m_to(m):
    return (m >> 6) & 63


@njit(cache=False, inline="always")
def m_flag(m):
    return (m >> 12) & 15


@njit(cache=False, inline="always")
def m_is_cap(m):
    f = m_flag(m)
    return f == CAP or f == EP_FLAG or f >= PNC


@njit(cache=False, inline="always")
def m_is_promo(m):
    return m_flag(m) >= PN


@njit(cache=False, inline="always")
def m_promo_pt(m):
    f = m_flag(m)
    if f == PN or f == PNC:
        return 1
    if f == PB or f == PBC:
        return 2
    if f == PR or f == PRC:
        return 3
    if f == PQ or f == PQC:
        return 4
    return 0


# --- attack queries -----------------------------------------------------------


@njit(cache=False)
def attackers_to(bb, mb, sq, by, occ):
    """Bitboard of `by`-coloured pieces attacking `sq` given occupancy `occ`."""
    base = by * 6
    a = U(0)
    a |= PAWN_ATT[1 - by][sq] & bb[0] & bb[6 + by]
    a |= KNIGHT_ATT[sq] & bb[1] & bb[6 + by]
    a |= KING_ATT[sq] & bb[5] & bb[6 + by]
    bishops = (bb[2] | bb[4]) & bb[6 + by]
    a |= bishop_attacks(sq, occ) & bishops
    rooks = (bb[3] | bb[4]) & bb[6 + by]
    a |= rook_attacks(sq, occ) & rooks
    _ = base
    return a


@njit(cache=False)
def attack_map(bb, mb, by, occ):
    """All squares attacked by side `by`."""
    a = U(0)
    p = bb[0] & bb[6 + by]
    while p != U(0):
        s = lsb(p)
        p &= p - ONE
        a |= PAWN_ATT[by][s]
    for pt in range(1, 6):
        x = bb[pt] & bb[6 + by]
        while x != U(0):
            s = lsb(x)
            x &= x - ONE
            if pt == 1:
                a |= KNIGHT_ATT[s]
            elif pt == 2:
                a |= bishop_attacks(s, occ)
            elif pt == 3:
                a |= rook_attacks(s, occ)
            elif pt == 4:
                a |= queen_attacks(s, occ)
            else:
                a |= KING_ATT[s]
    return a


@njit(cache=False)
def pinned_pieces(bb, mb, ksq, us, occ):
    them = 1 - us
    pin = U(0)
    bishops = (bb[2] | bb[4]) & bb[6 + them]
    rooks = (bb[3] | bb[4]) & bb[6 + them]
    snipers = (bishop_attacks(ksq, U(0)) & bishops) | (rook_attacks(ksq, U(0)) & rooks)
    while snipers != U(0):
        s = lsb(snipers)
        snipers &= snipers - ONE
        blockers = BETWEEN[ksq][s] & occ
        one_blocker = blockers != U(0) and (blockers & (blockers - ONE)) == U(0)
        if one_blocker and (blockers & bb[6 + us]) != U(0):
            pin |= blockers
    return pin


@njit(cache=False, inline="always")
def piece_at(mb, sq):
    return mb[sq]


# --- legal move generation --------------------------------------------------

WK_PATH = (ONE << U(5)) | (ONE << U(6))
WQ_PATH = (ONE << U(1)) | (ONE << U(2)) | (ONE << U(3))
BK_PATH = (ONE << U(61)) | (ONE << U(62))
BQ_PATH = (ONE << U(57)) | (ONE << U(58)) | (ONE << U(59))
# squares that must be free of enemy attack to castle (king start + path)
WK_SAFE = (ONE << U(4)) | WK_PATH
WQ_SAFE = (ONE << U(4)) | (ONE << U(3)) | (ONE << U(2))
BK_SAFE = (ONE << U(60)) | BK_PATH
BQ_SAFE = (ONE << U(60)) | (ONE << U(59)) | (ONE << U(58))
FILE_BB = np.array([FILE_A << U(f) for f in range(8)], np.uint64)
RANK_BB = np.array([RANK_1 << U(8 * r) for r in range(8)], np.uint64)
ADJ_FILES = np.array(
    [((int(FILE_BB[f - 1]) if f > 0 else 0) | (int(FILE_BB[f + 1]) if f < 7 else 0))
     for f in range(8)],
    np.uint64,
)


def _passed_masks():
    w = np.zeros(64, np.uint64)
    b = np.zeros(64, np.uint64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        span = int(FILE_BB[f]) | int(ADJ_FILES[f])
        ahead_w = 0
        for rr in range(r + 1, 8):
            ahead_w |= int(RANK_BB[rr])
        ahead_b = 0
        for rr in range(r):
            ahead_b |= int(RANK_BB[rr])
        w[sq] = span & ahead_w
        b[sq] = span & ahead_b
    return w, b


PASSED_W, PASSED_B = _passed_masks()


@njit(cache=False, inline="always")
def in_check(bb, mb):
    us = I(bb[STM])
    ksq = lsb(bb[5] & bb[6 + us])
    return attackers_to(bb, mb, ksq, 1 - us, bb[OCC]) != U(0)


@njit(cache=False, inline="always")
def _add_pawn(out, n, frm, to, cap, promo_rank):
    if (to >> 3) == promo_rank:
        if cap:
            out[n] = mk_move(frm, to, PQC); out[n + 1] = mk_move(frm, to, PRC)
            out[n + 2] = mk_move(frm, to, PBC); out[n + 3] = mk_move(frm, to, PNC)
        else:
            out[n] = mk_move(frm, to, PQ); out[n + 1] = mk_move(frm, to, PR)
            out[n + 2] = mk_move(frm, to, PB); out[n + 3] = mk_move(frm, to, PN)
        return n + 4
    out[n] = mk_move(frm, to, CAP if cap else QUIET)
    return n + 1


@njit(cache=False)
def gen_moves(bb, mb, out, caps_only):
    us = I(bb[STM])
    them = 1 - us
    occ = bb[OCC]
    own = bb[6 + us]
    enemy = bb[6 + them]
    ksq = lsb(bb[5] & own)
    n = 0

    checkers = attackers_to(bb, mb, ksq, them, occ)
    ncheck = popcount(checkers)

    occ_no_k = occ ^ (ONE << U(ksq))
    danger = attack_map(bb, mb, them, occ_no_k)

    kt = KING_ATT[ksq] & ~own & ~danger
    if caps_only:
        kt &= enemy
    x = kt
    while x != U(0):
        to = lsb(x); x &= x - ONE
        out[n] = mk_move(ksq, to, CAP if (enemy >> U(to)) & ONE else QUIET); n += 1

    if ncheck >= 2:
        return n

    if ncheck == 1:
        csq = lsb(checkers)
        cpt = mb[csq] % 6
        if cpt == 2 or cpt == 3 or cpt == 4:
            blk = BETWEEN[ksq][csq]
        else:
            blk = U(0)
        target = blk | checkers
    else:
        target = FULL_BB

    pin = pinned_pieces(bb, mb, ksq, us, occ)

    up = 8 if us == 0 else -8
    promo_rank = 7 if us == 0 else 0
    start_rank = 1 if us == 0 else 6

    pawns = bb[0] & own
    p = pawns
    while p != U(0):
        frm = lsb(p); p &= p - ONE
        fb = ONE << U(frm)
        if (pin & fb) != U(0):
            pline = LINE[ksq][frm]
        else:
            pline = FULL_BB
        # single / double push
        one = frm + up
        if 0 <= one < 64 and ((occ >> U(one)) & ONE) == U(0) and ((pline >> U(one)) & ONE) != U(0):
            if ((target >> U(one)) & ONE) != U(0) and not caps_only:
                n = _add_pawn(out, n, frm, one, False, promo_rank)
            elif ((target >> U(one)) & ONE) != U(0) and (one >> 3) == promo_rank:
                n = _add_pawn(out, n, frm, one, False, promo_rank)
            if (frm >> 3) == start_rank:
                two = one + up
                if ((occ >> U(two)) & ONE) == U(0) and ((target >> U(two)) & ONE) != U(0) \
                        and ((pline >> U(two)) & ONE) != U(0) and not caps_only:
                    out[n] = mk_move(frm, two, DPUSH); n += 1
        # captures
        caps = PAWN_ATT[us][frm] & enemy & target & pline
        while caps != U(0):
            to = lsb(caps); caps &= caps - ONE
            n = _add_pawn(out, n, frm, to, True, promo_rank)
        # en passant
        ep = I(bb[EP])
        can_ep = ep != 64 and (PAWN_ATT[us][frm] & (ONE << U(ep))) != U(0)
        if can_ep and ((pline >> U(ep)) & ONE) != U(0):
            capsq = ep - up
            occ2 = (occ ^ fb ^ (ONE << U(capsq))) | (ONE << U(ep))
            bishops = (bb[2] | bb[4]) & enemy
            rooks = (bb[3] | bb[4]) & enemy
            b_ok = (bishop_attacks(ksq, occ2) & bishops) == U(0)
            if b_ok and (rook_attacks(ksq, occ2) & rooks) == U(0):
                out[n] = mk_move(frm, ep, EP_FLAG); n += 1

    # knights
    kn = bb[1] & own & ~pin
    while kn != U(0):
        frm = lsb(kn); kn &= kn - ONE
        tt = KNIGHT_ATT[frm] & ~own & target
        if caps_only:
            tt &= enemy
        while tt != U(0):
            to = lsb(tt); tt &= tt - ONE
            out[n] = mk_move(frm, to, CAP if (enemy >> U(to)) & ONE else QUIET); n += 1

    # sliders
    for pt in range(2, 5):
        b2 = bb[pt] & own
        while b2 != U(0):
            frm = lsb(b2); b2 &= b2 - ONE
            if pt == 2:
                tt = bishop_attacks(frm, occ)
            elif pt == 3:
                tt = rook_attacks(frm, occ)
            else:
                tt = queen_attacks(frm, occ)
            tt &= ~own & target
            if caps_only:
                tt &= enemy
            if (pin & (ONE << U(frm))) != U(0):
                tt &= LINE[ksq][frm]
            while tt != U(0):
                to = lsb(tt); tt &= tt - ONE
                out[n] = mk_move(frm, to, CAP if (enemy >> U(to)) & ONE else QUIET); n += 1

    # castling
    if not caps_only and ncheck == 0:
        cr = I(bb[CAST])
        if us == 0:
            if (cr & WK) and (occ & WK_PATH) == U(0) and (danger & WK_SAFE) == U(0):
                out[n] = mk_move(4, 6, CK); n += 1
            if (cr & WQ) and (occ & WQ_PATH) == U(0) and (danger & WQ_SAFE) == U(0):
                out[n] = mk_move(4, 2, CQ); n += 1
        else:
            if (cr & BK) and (occ & BK_PATH) == U(0) and (danger & BK_SAFE) == U(0):
                out[n] = mk_move(60, 62, CK); n += 1
            if (cr & BQ) and (occ & BQ_PATH) == U(0) and (danger & BQ_SAFE) == U(0):
                out[n] = mk_move(60, 58, CQ); n += 1

    return n


# --- zobrist ----------------------------------------------------------------
_zr = np.random.RandomState(0xBEEF)
ZPSQ = _zr.randint(0, 1 << 63, size=(12, 64), dtype=np.uint64) | (
    _zr.randint(0, 1 << 63, size=(12, 64), dtype=np.uint64) << np.uint64(1)
)
ZCAST = _zr.randint(0, 1 << 63, size=16, dtype=np.uint64)
ZEP = _zr.randint(0, 1 << 63, size=8, dtype=np.uint64)
ZSTM = np.uint64(_zr.randint(0, 1 << 63, dtype=np.uint64))

# undo record: hist[ply, 0..] = move, captured_code(+1, 0=none), castling, ep, halfmove, key
HIST_W = 6


@njit(cache=False, inline="always")
def _put(bb, mb, code, sq):
    pt = code % 6
    col = code // 6
    m = ONE << U(sq)
    bb[pt] |= m
    bb[6 + col] |= m
    bb[OCC] |= m
    mb[sq] = code
    bb[KEY] ^= ZPSQ[code, sq]


@njit(cache=False, inline="always")
def _rm(bb, mb, code, sq):
    pt = code % 6
    col = code // 6
    m = ~(ONE << U(sq))
    bb[pt] &= m
    bb[6 + col] &= m
    bb[OCC] &= m
    mb[sq] = -1
    bb[KEY] ^= ZPSQ[code, sq]


@njit(cache=False, inline="always")
def _move_pc(bb, mb, code, frm, to):
    _rm(bb, mb, code, frm)
    _put(bb, mb, code, to)


_CASTLE_MASK = np.full(64, 15, np.int64)
_CASTLE_MASK[4] = 15 ^ (WK | WQ)
_CASTLE_MASK[0] = 15 ^ WQ
_CASTLE_MASK[7] = 15 ^ WK
_CASTLE_MASK[60] = 15 ^ (BK | BQ)
_CASTLE_MASK[56] = 15 ^ BQ
_CASTLE_MASK[63] = 15 ^ BK


@njit(cache=False)
def make_move(bb, mb, hist, ply, m):
    us = I(bb[STM])
    them = 1 - us
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    moving = mb[frm]
    pt = moving % 6

    hist[ply, 0] = m
    hist[ply, 2] = bb[CAST]
    hist[ply, 3] = bb[EP]
    hist[ply, 4] = bb[HALF]
    hist[ply, 5] = bb[KEY]

    old_ep = I(bb[EP])
    if old_ep != 64:
        bb[KEY] ^= ZEP[old_ep & 7]
    bb[EP] = U(64)
    bb[HALF] += U(1)
    captured = 0

    if flag == EP_FLAG:
        capsq = to - 8 if us == 0 else to + 8
        _rm(bb, mb, 6 * them + 0, capsq)
        _move_pc(bb, mb, moving, frm, to)
        captured = 1
        bb[HALF] = U(0)
    elif flag == CK or flag == CQ:
        _move_pc(bb, mb, moving, frm, to)
        if flag == CK:
            rf = to + 1; rt = to - 1
        else:
            rf = to - 2; rt = to + 1
        _move_pc(bb, mb, 6 * us + 3, rf, rt)
    else:
        tgt = mb[to]
        if tgt >= 0:
            _rm(bb, mb, tgt, to)
            captured = tgt + 1
            bb[HALF] = U(0)
        promo = m_promo_pt(m)
        if promo != 0:
            _rm(bb, mb, moving, frm)
            _put(bb, mb, 6 * us + promo, to)
            bb[HALF] = U(0)
        else:
            _move_pc(bb, mb, moving, frm, to)
        if pt == 0:
            bb[HALF] = U(0)
            if flag == DPUSH:
                ep = frm + 8 if us == 0 else frm - 8
                bb[EP] = U(ep)
                bb[KEY] ^= ZEP[ep & 7]

    hist[ply, 1] = captured

    new_cast = I(bb[CAST]) & _CASTLE_MASK[frm] & _CASTLE_MASK[to]
    if new_cast != I(bb[CAST]):
        bb[KEY] ^= ZCAST[I(bb[CAST])]
        bb[KEY] ^= ZCAST[new_cast]
        bb[CAST] = U(new_cast)

    if us == 1:
        bb[FULL] += U(1)
    bb[STM] = U(them)
    bb[KEY] ^= ZSTM


@njit(cache=False)
def unmake_move(bb, mb, hist, ply):
    m = np.int32(hist[ply, 0])
    us = 1 - I(bb[STM])
    them = 1 - us
    frm = m_from(m)
    to = m_to(m)
    flag = m_flag(m)
    captured = I(hist[ply, 1])

    bb[STM] = U(us)
    if us == 1:
        bb[FULL] -= U(1)

    if flag == EP_FLAG:
        moving = 6 * us + 0
        _rm(bb, mb, moving, to)
        _put(bb, mb, moving, frm)
        capsq = to - 8 if us == 0 else to + 8
        _put(bb, mb, 6 * them + 0, capsq)
    elif flag == CK or flag == CQ:
        moving = mb[to]
        _rm(bb, mb, moving, to)
        _put(bb, mb, moving, frm)
        if flag == CK:
            rf = to + 1; rt = to - 1
        else:
            rf = to - 2; rt = to + 1
        _rm(bb, mb, 6 * us + 3, rt)
        _put(bb, mb, 6 * us + 3, rf)
    else:
        promo = m_promo_pt(m)
        if promo != 0:
            _rm(bb, mb, 6 * us + promo, to)
            _put(bb, mb, 6 * us + 0, frm)
        else:
            moving = mb[to]
            _rm(bb, mb, moving, to)
            _put(bb, mb, moving, frm)
        if captured != 0:
            _put(bb, mb, captured - 1, to)

    bb[CAST] = hist[ply, 2]
    bb[EP] = hist[ply, 3]
    bb[HALF] = hist[ply, 4]
    bb[KEY] = hist[ply, 5]


@njit(cache=False)
def make_null(bb, hist, ply):
    hist[ply, 0] = 0
    hist[ply, 2] = bb[CAST]
    hist[ply, 3] = bb[EP]
    hist[ply, 4] = bb[HALF]
    hist[ply, 5] = bb[KEY]
    if I(bb[EP]) != 64:
        bb[KEY] ^= ZEP[I(bb[EP]) & 7]
    bb[EP] = U(64)
    bb[STM] = U(1 - I(bb[STM]))
    bb[KEY] ^= ZSTM
    bb[HALF] += U(1)


@njit(cache=False)
def unmake_null(bb, hist, ply):
    bb[STM] = U(1 - I(bb[STM]))
    bb[EP] = hist[ply, 3]
    bb[HALF] = hist[ply, 4]
    bb[KEY] = hist[ply, 5]


@njit(cache=False)
def perft(bb, mb, hist, ply, depth):
    if depth == 0:
        return 1
    out = np.empty(256, np.int32)
    n = gen_moves(bb, mb, out, False)
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        make_move(bb, mb, hist, ply, out[i])
        total += perft(bb, mb, hist, ply + 1, depth - 1)
        unmake_move(bb, mb, hist, ply)
    return total


# --- Python side: FEN <-> arrays -------------------------------------------

_FEN_PIECE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
              "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}


def fen_to_arrays(fen: str):
    bb = np.zeros(BB_LEN, np.uint64)
    mb = np.full(64, -1, np.int8)
    parts = fen.split()
    rows = parts[0].split("/")
    for r, row in enumerate(rows):
        rank = 7 - r
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                code = _FEN_PIECE[ch]
                sq = rank * 8 + file
                mb[sq] = code
                pt, col = code % 6, code // 6
                bb[pt] |= np.uint64(1) << np.uint64(sq)
                bb[6 + col] |= np.uint64(1) << np.uint64(sq)
                file += 1
    bb[OCC] = bb[WOCC] | bb[BOCC]
    bb[STM] = 0 if parts[1] == "w" else 1
    rights = parts[2] if len(parts) > 2 else "-"
    cr = 0
    if "K" in rights:
        cr |= WK
    if "Q" in rights:
        cr |= WQ
    if "k" in rights:
        cr |= BK
    if "q" in rights:
        cr |= BQ
    bb[CAST] = cr
    ep = parts[3] if len(parts) > 3 else "-"
    bb[EP] = 64 if ep == "-" else (ord(ep[0]) - 97) + 8 * (int(ep[1]) - 1)
    bb[HALF] = int(parts[4]) if len(parts) > 4 else 0
    bb[FULL] = int(parts[5]) if len(parts) > 5 else 1
    bb[KEY] = _zobrist_of(bb, mb)
    return bb, mb


def _zobrist_of(bb, mb):
    k = np.uint64(0)
    for sq in range(64):
        c = mb[sq]
        if c >= 0:
            k ^= ZPSQ[c, sq]
    k ^= ZCAST[int(bb[CAST])]
    if int(bb[EP]) != 64:
        k ^= ZEP[int(bb[EP]) & 7]
    if int(bb[STM]) == 1:
        k ^= ZSTM
    return k


_UCI_PROMO = {1: "n", 2: "b", 3: "r", 4: "q"}


def move_to_uci(m: int) -> str:
    frm = m & 63
    to = (m >> 6) & 63
    s = chr(97 + (frm & 7)) + str(1 + (frm >> 3)) + chr(97 + (to & 7)) + str(1 + (to >> 3))
    p = (m >> 12) & 15
    if p >= PN:
        s += _UCI_PROMO[m_promo_pt_py(m)]
    return s


def m_promo_pt_py(m):
    f = (m >> 12) & 15
    return {PN: 1, PNC: 1, PB: 2, PBC: 2, PR: 3, PRC: 3, PQ: 4, PQC: 4}.get(f, 0)


def legal_move_ucis(fen: str):
    bb, mb = fen_to_arrays(fen)
    out = np.empty(256, np.int32)
    n = gen_moves(bb, mb, out, False)
    return sorted(move_to_uci(int(out[i])) for i in range(n))


def perft_fen(fen: str, depth: int) -> int:
    bb, mb = fen_to_arrays(fen)
    hist = np.zeros((MAX_PLY, HIST_W), np.uint64)
    return int(perft(bb, mb, hist, 0, depth))


# --- evaluation -----------------------------------------------------------
# PeSTO tapered material + piece-square tables (White POV, a1..h8).
MG_VAL = np.array([82, 337, 365, 477, 1025, 0], np.int64)
EG_VAL = np.array([94, 281, 297, 512, 936, 0], np.int64)
_PHW = np.array([0, 1, 1, 2, 4, 0], np.int64)
PHASE_MAX = 24

_MG_PST = np.array([
 [0,0,0,0,0,0,0,0,-35,-1,-20,-23,-15,24,38,-22,-26,-4,-4,-10,3,3,33,-12,-27,-2,-5,12,17,6,10,-25,
  -14,13,6,21,23,12,17,-23,-6,7,26,31,65,56,25,-20,98,134,61,95,68,126,34,-11,0,0,0,0,0,0,0,0],
 [-105,-21,-58,-33,-17,-28,-19,-23,-29,-53,-12,-3,-1,18,-14,-19,-23,-9,12,10,19,17,25,-16,
  -13,4,16,13,28,19,21,-8,-9,17,19,53,37,69,18,22,-47,60,37,65,84,129,73,44,
  -73,-41,72,36,23,62,7,-17,-167,-89,-34,-49,61,-97,-15,-107],
 [-33,-3,-14,-21,-13,-12,-39,-21,4,15,16,0,7,21,33,1,0,15,15,15,14,27,18,10,
  -6,13,13,26,34,12,10,4,-4,5,19,50,37,37,7,-2,-16,37,43,40,35,50,37,-2,
  -26,16,-18,-13,30,59,18,-47,-29,4,-82,-37,-25,-42,7,-8],
 [-19,-13,1,17,16,7,-37,-26,-44,-16,-20,-9,-1,11,-6,-71,-45,-25,-16,-17,3,0,-5,-33,
  -36,-26,-12,-1,9,-7,6,-23,-24,-11,7,26,24,35,-8,-20,-5,19,26,36,17,45,61,16,
  27,32,58,62,80,67,26,44,32,42,32,51,63,9,31,43],
 [-1,-18,-9,10,-15,-25,-31,-50,-35,-8,11,2,8,15,-3,1,-14,2,-11,-2,-5,2,14,5,
  -9,-26,-9,-10,-2,-4,3,-3,-27,-27,-16,-16,-1,17,-2,1,-13,-17,7,8,29,56,47,57,
  -24,-39,-5,1,-16,57,28,54,-28,0,29,12,59,44,43,45],
 [-15,36,12,-54,8,-28,24,14,1,7,-8,-64,-43,-16,9,8,-14,-14,-22,-46,-44,-30,-15,-27,
  -49,-1,-27,-39,-46,-44,-33,-51,-17,-20,-12,-27,-30,-25,-14,-36,-9,24,2,-16,-20,6,22,-22,
  29,-1,-20,-7,-8,-4,-38,-29,-65,23,16,-15,-56,-34,2,13],
], np.int64)
_EG_PST = np.array([
 [0,0,0,0,0,0,0,0,13,8,8,10,13,0,2,-7,4,7,-6,1,0,-5,-1,-8,13,9,-3,-7,-7,-8,3,-1,
  32,24,13,5,-2,4,17,17,94,100,85,67,56,53,82,84,178,173,158,134,147,132,165,187,0,0,0,0,0,0,0,0],
 [-29,-51,-23,-15,-22,-18,-50,-64,-42,-20,-10,-5,-2,-20,-23,-44,-23,-3,-1,15,10,-3,-20,-22,
  -18,-6,16,25,16,17,4,-18,-17,3,22,22,22,11,8,-18,-24,-20,10,9,-1,-9,-19,-41,
  -25,-8,-25,-2,-9,-25,-24,-52,-58,-38,-13,-28,-31,-27,-63,-99],
 [-23,-9,-23,-5,-9,-16,-5,-17,-14,-18,-7,-1,4,-9,-15,-27,-12,-3,8,10,13,3,-7,-15,
  -6,3,13,19,7,10,-3,-9,-3,9,12,9,14,10,3,2,2,-8,0,-1,-2,6,0,4,
  -8,-4,7,-12,-3,-13,-4,-14,-14,-21,-11,-8,-7,-9,-17,-24],
 [-9,2,3,-1,-5,-13,4,-20,-6,-6,0,2,-9,-9,-11,-3,-4,0,-5,-1,-7,-12,-8,-16,
  3,5,8,4,-5,-6,-8,-11,4,3,13,1,2,1,-1,2,7,7,7,5,4,-3,-5,-3,
  11,13,13,11,-3,3,8,3,13,10,18,15,12,12,8,5],
 [-33,-28,-22,-43,-5,-32,-20,-41,-22,-23,-30,-16,-16,-23,-36,-32,-16,-27,15,6,9,17,10,5,
  -18,28,19,47,31,34,39,23,3,22,24,45,57,40,57,36,-20,6,9,49,47,35,19,9,
  -17,20,32,41,58,25,30,0,-9,22,22,27,27,19,10,20],
 [-53,-34,-21,-11,-28,-14,-24,-43,-27,-11,4,13,14,4,-5,-17,-19,-3,11,21,23,16,7,-9,
  -18,-4,21,24,27,23,9,-11,-8,22,24,27,26,33,26,3,10,17,23,15,20,45,44,13,
  -12,17,14,17,17,38,23,11,-74,-35,-18,-18,-11,15,4,-17],
], np.int64)

MATE = 30000
MATE_IN_MAX = MATE - 512
INF = MATE + 1

_PASS_MG = np.array([0, 5, 10, 15, 30, 55, 90, 0], np.int64)
_PASS_EG = np.array([0, 10, 18, 30, 55, 95, 160, 0], np.int64)


@njit(cache=False)
def evaluate(bb, mb):
    occ = bb[OCC]
    ph = 0
    for pt in range(1, 5):
        ph += _PHW[pt] * popcount(bb[pt])
    if ph > PHASE_MAX:
        ph = PHASE_MAX

    mg = 0
    eg = 0
    mob = np.zeros(2, np.int64)
    danger = np.zeros(2, np.int64)

    for col in range(2):
        sign = 1 if col == 0 else -1
        them = 1 - col
        ksq_them = lsb(bb[5] & bb[6 + them])
        ring = KING_ATT[ksq_them] | (ONE << U(ksq_them))
        own = bb[6 + col]
        for pt in range(6):
            x = bb[pt] & own
            while x != U(0):
                sq = lsb(x); x &= x - ONE
                idx = sq if col == 0 else (sq ^ 56)
                mg += sign * (MG_VAL[pt] + _MG_PST[pt, idx])
                eg += sign * (EG_VAL[pt] + _EG_PST[pt, idx])
                if 1 <= pt <= 4:
                    if pt == 1:
                        att = KNIGHT_ATT[sq]
                    elif pt == 2:
                        att = bishop_attacks(sq, occ)
                    elif pt == 3:
                        att = rook_attacks(sq, occ)
                    else:
                        att = queen_attacks(sq, occ)
                    att &= ~own
                    mob[col] += 2 * popcount(att)
                    rh = popcount(att & ring)
                    if rh > 0:
                        w = 2 if pt <= 2 else (3 if pt == 3 else 5)
                        danger[them] += w * rh
        # bishop pair
        if popcount(bb[2] & own) >= 2:
            mg += sign * 25
            eg += sign * 25
        # pawns
        own_p = bb[0] & own
        enemy_p = bb[0] & bb[6 + them]
        pp = own_p
        while pp != U(0):
            sq = lsb(pp); pp &= pp - ONE
            f = sq & 7
            fmask = FILE_BB[f]
            if (own_p & ADJ_FILES[f]) == U(0):
                mg -= sign * 12; eg -= sign * 12
            if popcount(own_p & fmask) > 1:
                mg -= sign * 5; eg -= sign * 10
            span = PASSED_W[sq] if col == 0 else PASSED_B[sq]
            if (span & enemy_p) == U(0):
                rr_idx = (sq >> 3) if col == 0 else (7 - (sq >> 3))
                mg += sign * _PASS_MG[rr_idx]
                eg += sign * _PASS_EG[rr_idx]
        # rooks open files
        rr2 = bb[3] & own
        while rr2 != U(0):
            sq = lsb(rr2); rr2 &= rr2 - ONE
            fmask = FILE_BB[sq & 7]
            if (fmask & bb[0]) == U(0):
                mg += sign * 22
            elif (fmask & own_p) == U(0):
                mg += sign * 10

    wd = _king_danger(danger[0])
    bd = _king_danger(danger[1])
    mg += bd - wd
    mg += mob[0] - mob[1]

    # truncate toward zero so the tapered score is exactly colour-symmetric
    num = mg * ph + eg * (PHASE_MAX - ph)
    score = num // PHASE_MAX if num >= 0 else -((-num) // PHASE_MAX)
    stm = score if I(bb[STM]) == 0 else -score
    return stm + 14


@njit(cache=False, inline="always")
def _king_danger(units):
    u = units if units < 40 else 40
    return (u * u * 3) // 8


# --- search -------------------------------------------------------------------
# numba treats module globals as read-only, so the mutable search state is
# passed explicitly to every node function as a fixed bundle:
#   bb   uint64[16]   position (mutated by make/unmake)
#   mb   int8[64]     mailbox
#   tt   int64[N,4]   transposition table: [key, move, score, depth*4+flag]
#   gh   uint64[H,6]  make/unmake undo stack, seeded with real game history
#   kl   int32[P,2]   killer moves per ply
#   hi   int32[2,64,64] history heuristic
#   mv   int32[P,256] scratch move buffers per ply
#   ct   int64[8]     [nodes, stop, max_nodes, seldepth, game_ply, .., .., ..]

_TT_BITS = 22
_TT_SIZE = 1 << _TT_BITS
_TT_MASK = _TT_SIZE - 1
N_NODES, N_STOP, N_MAXN, N_SELD, N_GPLY = 0, 1, 2, 3, 4

MATE_S = MATE
MIMAX = MATE_IN_MAX
INF_S = INF


@njit(cache=False, inline="always")
def _tt_get(tt, key):
    i = (I(key) & _TT_MASK)
    if tt[i, 3] != 0 and tt[i, 0] == I(key):
        return i
    return -1


@njit(cache=False, inline="always")
def _tt_put(tt, key, mv, score, depth, flag):
    i = (I(key) & _TT_MASK)
    packed_old = tt[i, 3]
    old_depth = (packed_old >> 2) - 128 if packed_old != 0 else -128
    if packed_old == 0 or tt[i, 0] == I(key) or depth >= old_depth - 2:
        keep = tt[i, 1] if (mv == 0 and tt[i, 0] == I(key)) else mv
        tt[i, 0] = I(key)
        tt[i, 1] = keep
        tt[i, 2] = score
        tt[i, 3] = ((depth + 128) << 2) | flag


@njit(cache=False, inline="always")
def _rep_or_50(bb, gh, ct, ply):
    if I(bb[HALF]) >= 100:
        return True
    hm = I(bb[HALF])
    total = ct[N_GPLY] + ply
    back = 4
    while back <= hm and back <= total:
        if gh[total - back, 5] == bb[KEY]:
            return True
        back += 2
    return False


@njit(cache=False, inline="always")
def _has_np(bb, col):
    return (bb[6 + col] & ~(bb[0] | bb[5])) != U(0)


@njit(cache=False)
def _order(bb, mb, kl, hi, buf, n, ply, tt_move):
    stm = I(bb[STM])
    k0 = kl[ply, 0]
    k1 = kl[ply, 1]
    sc = np.empty(n, np.int64)
    for i in range(n):
        m = buf[i]
        if m == tt_move:
            v = 1 << 24
        elif m_is_cap(m):
            to = m_to(m)
            victim = mb[to] % 6 if mb[to] >= 0 else 0
            attacker = mb[m_from(m)] % 6
            v = (1 << 20) + victim * 16 - attacker
        elif m == k0:
            v = 1 << 19
        elif m == k1:
            v = (1 << 19) - 1
        else:
            v = hi[stm, m_from(m), m_to(m)]
        sc[i] = v
    for i in range(1, n):
        a = sc[i]
        mm = buf[i]
        j = i - 1
        while j >= 0 and sc[j] < a:
            sc[j + 1] = sc[j]
            buf[j + 1] = buf[j]
            j -= 1
        sc[j + 1] = a
        buf[j + 1] = mm


@njit(cache=False)
def _qs(bb, mb, tt, gh, kl, hi, mv, ct, ply, alpha, beta):
    ct[N_NODES] += 1
    if ct[N_NODES] >= ct[N_MAXN]:
        ct[N_STOP] = 1
        return 0
    checked = in_check(bb, mb)
    if not checked:
        stand = evaluate(bb, mb)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
    else:
        stand = -INF_S
    if ply >= MAX_PLY - 1:
        return stand if not checked else evaluate(bb, mb)

    buf = mv[ply]
    n = gen_moves(bb, mb, buf, not checked)
    if n == 0:
        return -MATE_S + ply if checked else stand
    _order(bb, mb, kl, hi, buf, n, ply, 0)

    best = stand
    gp = ct[N_GPLY]
    for i in range(n):
        m = buf[i]
        if not checked and not m_is_promo(m):
            to = m_to(m)
            victim = MG_VAL[mb[to] % 6] if mb[to] >= 0 else 100
            attacker = MG_VAL[mb[m_from(m)] % 6]
            if victim + 90 < attacker and stand + victim + 150 < alpha:
                continue
        make_move(bb, mb, gh, gp + ply, m)
        s = -_qs(bb, mb, tt, gh, kl, hi, mv, ct, ply + 1, -beta, -alpha)
        unmake_move(bb, mb, gh, gp + ply)
        if ct[N_STOP] == 1:
            return 0
        if s > best:
            best = s
            if s > alpha:
                alpha = s
                if alpha >= beta:
                    break
    return best


@njit(cache=False)
def _nm(bb, mb, tt, gh, kl, hi, mv, ct, depth, ply, alpha, beta, is_pv):
    if ct[N_STOP] == 1:
        return 0
    ct[N_NODES] += 1
    if ct[N_NODES] >= ct[N_MAXN]:
        ct[N_STOP] = 1
        return 0
    if ply > ct[N_SELD]:
        ct[N_SELD] = ply

    if ply > 0 and _rep_or_50(bb, gh, ct, ply):
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate(bb, mb)

    if alpha < -MATE_S + ply:
        alpha = -MATE_S + ply
    if beta > MATE_S - ply - 1:
        beta = MATE_S - ply - 1
    if alpha >= beta:
        return alpha

    checked = in_check(bb, mb)
    if checked:
        depth += 1
    if depth <= 0:
        return _qs(bb, mb, tt, gh, kl, hi, mv, ct, ply, alpha, beta)

    tt_move = np.int32(0)
    ti = _tt_get(tt, bb[KEY])
    if ti >= 0:
        tt_move = np.int32(tt[ti, 1])
        td = (tt[ti, 3] >> 2) - 128
        if (not is_pv) and td >= depth:
            s = tt[ti, 2]
            if s >= MIMAX:
                s -= ply
            elif s <= -MIMAX:
                s += ply
            fl = tt[ti, 3] & 3
            if fl == 1:
                return s
            if fl == 2 and s >= beta:
                return s
            if fl == 3 and s <= alpha:
                return s

    static = -INF_S if checked else evaluate(bb, mb)

    can_rfp = (not is_pv) and (not checked) and depth <= 6 and abs(beta) < MIMAX
    if can_rfp and static - 80 * depth >= beta:
        return static

    if (not is_pv) and (not checked) and depth >= 3 and static >= beta and _has_np(bb, I(bb[STM])):
        r = 3 + depth // 4
        make_null(bb, gh, ct[N_GPLY] + ply)
        s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, depth - r, ply + 1, -beta, -beta + 1, False)
        unmake_null(bb, gh, ct[N_GPLY] + ply)
        if s >= beta and abs(s) < MIMAX:
            return beta

    buf = mv[ply]
    n = gen_moves(bb, mb, buf, False)
    if n == 0:
        return -MATE_S + ply if checked else 0
    _order(bb, mb, kl, hi, buf, n, ply, tt_move)

    old_alpha = alpha
    best = -INF_S
    best_move = np.int32(0)
    stm = I(bb[STM])
    quiets = np.empty(64, np.int32)
    nq = 0
    gp = ct[N_GPLY]

    for i in range(n):
        m = buf[i]
        quiet = (not m_is_cap(m)) and (not m_is_promo(m))

        if (not is_pv) and (not checked) and quiet and best > -MIMAX:
            if depth <= 4 and i >= 4 + depth * depth:
                continue
            if depth <= 3 and static + 90 * depth <= alpha and i > 0:
                continue

        make_move(bb, mb, gh, gp + ply, m)
        gives_check = in_check(bb, mb)
        nd = depth - 1
        if i == 0:
            s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, nd, ply + 1, -beta, -alpha, is_pv)
        else:
            r = 0
            if depth >= 3 and quiet and (not gives_check) and (not checked):
                r = 1 + (1 if i >= 6 else 0) + (1 if (depth >= 6 and i >= 12) else 0)
                if is_pv:
                    r -= 1
                if r < 0:
                    r = 0
                if r > nd - 1:
                    r = nd - 1
            s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, nd - r, ply + 1, -alpha - 1, -alpha, False)
            if s > alpha and r > 0:
                s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, nd, ply + 1, -alpha - 1, -alpha, False)
            if s > alpha and s < beta:
                s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, nd, ply + 1, -beta, -alpha, True)
        unmake_move(bb, mb, gh, gp + ply)

        if ct[N_STOP] == 1:
            return 0

        if s > best:
            best = s
            best_move = m
            if s > alpha:
                alpha = s
                if alpha >= beta:
                    if quiet:
                        if kl[ply, 0] != m:
                            kl[ply, 1] = kl[ply, 0]
                            kl[ply, 0] = m
                        hi[stm, m_from(m), m_to(m)] += depth * depth
                        for qi in range(nq):
                            qq = quiets[qi]
                            hi[stm, m_from(qq), m_to(qq)] -= depth * depth
                    break
        if quiet and nq < 64:
            quiets[nq] = m
            nq += 1

    if best <= old_alpha:
        fl = 3
    elif best >= beta:
        fl = 2
    else:
        fl = 1
    ss = best
    if ss >= MIMAX:
        ss += ply
    elif ss <= -MIMAX:
        ss -= ply
    _tt_put(tt, bb[KEY], best_move, ss, depth, fl)
    return best


@njit(cache=False)
def _root(bb, mb, tt, gh, kl, hi, mv, ct, depth, alpha, beta):
    buf = mv[0]
    n = gen_moves(bb, mb, buf, False)
    tt_move = np.int32(0)
    ti = _tt_get(tt, bb[KEY])
    if ti >= 0:
        tt_move = np.int32(tt[ti, 1])
    _order(bb, mb, kl, hi, buf, n, 0, tt_move)

    best = -INF_S
    best_move = buf[0]
    a = alpha
    gp = ct[N_GPLY]
    for i in range(n):
        m = buf[i]
        make_move(bb, mb, gh, gp, m)
        if i == 0:
            s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, depth - 1, 1, -beta, -a, True)
        else:
            s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, depth - 1, 1, -a - 1, -a, False)
            if s > a and s < beta:
                s = -_nm(bb, mb, tt, gh, kl, hi, mv, ct, depth - 1, 1, -beta, -a, True)
        unmake_move(bb, mb, gh, gp)
        if ct[N_STOP] == 1:
            break
        if s > best:
            best = s
            best_move = m
            if s > a:
                a = s
    if ct[N_STOP] == 0:
        fl = 2 if best >= beta else (1 if best > alpha else 3)
        _tt_put(tt, bb[KEY], best_move, best, depth, fl)
    return best_move, best


# --- Python driver ---------------------------------------------------------


class Engine:
    """Holds the compiled code and the transposition table; no per-move state."""

    def __init__(self):
        self._nps = 700_000.0
        self.tt = np.zeros((_TT_SIZE, 4), np.int64)
        self.gh = np.zeros((MAX_PLY + 1024, HIST_W), np.uint64)
        self.kl = np.zeros((MAX_PLY, 2), np.int32)
        self.hi = np.zeros((2, 64, 64), np.int32)
        self.mv = np.zeros((MAX_PLY, 256), np.int32)
        self.ct = np.zeros(8, np.int64)
        # warm the JIT (compiles the whole graph)
        bb, mb = fen_to_arrays("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.ct[N_MAXN] = 30_000
        self.ct[:] = [0, 0, 30_000, 0, 0, 0, 0, 0]
        _root(bb, mb, self.tt, self.gh, self.kl, self.hi, self.mv, self.ct, 4, -INF, INF)
        self.clear()

    def clear(self):
        self.tt[:, 3] = 0

    def best_move(self, fen, moves, soft_ms, hard_ms, max_depth=64):
        import time

        bb, mb = fen_to_arrays(fen)
        gp = _seed_history(bb, mb, list(moves), self.gh)
        self.kl[:, :] = 0
        self.hi[:, :, :] = 0
        self.ct[:] = 0
        self.ct[N_GPLY] = gp

        buf0 = np.empty(256, np.int32)
        if gen_moves(bb, mb, buf0, False) == 1:
            return move_to_uci(int(buf0[0])), 0, 1, 0

        start = time.perf_counter()
        self.ct[N_MAXN] = max(60_000, int(self._nps * (hard_ms / 1000.0) * 1.15))

        best = None
        score = 0
        completed = 0
        for depth in range(1, max_depth + 1):
            if depth >= 4 and abs(score) < MATE_IN_MAX:
                window = 30
                while True:
                    mvv, sc = _root(bb, mb, self.tt, self.gh, self.kl, self.hi,
                                    self.mv, self.ct, depth, score - window, score + window)
                    if self.ct[N_STOP] == 1:
                        break
                    if sc <= score - window or sc >= score + window:
                        window *= 3
                    else:
                        break
            else:
                mvv, sc = _root(bb, mb, self.tt, self.gh, self.kl, self.hi,
                                self.mv, self.ct, depth, -INF, INF)

            elapsed = time.perf_counter() - start
            if self.ct[N_STOP] == 1 and best is not None:
                break
            best = mvv
            score = int(sc)
            completed = depth
            if elapsed > 0 and self.ct[N_NODES] > 5000:
                self._nps = 0.6 * self._nps + 0.4 * (self.ct[N_NODES] / elapsed)
            if elapsed * 1000.0 >= soft_ms or abs(score) >= MATE_IN_MAX:
                break
            if elapsed * 1000.0 * 2.0 >= hard_ms:
                break

        return move_to_uci(int(best)), score, completed, int(self.ct[N_NODES])


def _seed_history(bb, mb, moves, gh):
    ply = 0
    for uci in moves:
        out = np.empty(256, np.int32)
        n = gen_moves(bb, mb, out, False)
        found = -1
        for i in range(n):
            if move_to_uci(int(out[i])) == uci:
                found = int(out[i])
                break
        if found < 0:
            break
        make_move(bb, mb, gh, ply, found)
        ply += 1
    return ply
