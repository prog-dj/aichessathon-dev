//! Static exchange evaluation, `see_ge`: does the capture on `mv.to()` win at
//! least `threshold` centipawns for the side to move? Classic iterative form.

use crate::position::Position;
use crate::tables::tables;
use crate::types::*;

const SEE_VAL: [i32; 7] = [100, 320, 330, 500, 900, 0, 0];

#[inline]
fn val(pt: PieceType) -> i32 {
    SEE_VAL[pt.index()]
}

pub fn see_ge(pos: &Position, mv: Move, threshold: i32) -> bool {
    if mv.is_castle() || mv.is_promotion() {
        return true; // don't bother; handled well enough by search
    }
    let from = mv.from();
    let to = mv.to();

    let captured = if mv.is_en_passant() {
        Some(PieceType::Pawn)
    } else {
        pos.mailbox[to as usize].map(|(_, p)| p)
    };

    let mut swap = captured.map(val).unwrap_or(0) - threshold;
    if swap < 0 {
        return false;
    }

    let mut next = pos.mailbox[from as usize].unwrap().1;
    swap = val(next) - swap;
    if swap <= 0 {
        return true;
    }

    let t = tables();
    let mut occ = pos.occupied() ^ bb(from) ^ bb(to);
    if mv.is_en_passant() {
        let cap = if pos.stm == Color::White { to - 8 } else { to + 8 };
        occ ^= bb(cap);
    }

    let bishops = pos.pieces[PieceType::Bishop.index()] | pos.pieces[PieceType::Queen.index()];
    let rooks = pos.pieces[PieceType::Rook.index()] | pos.pieces[PieceType::Queen.index()];

    let mut attackers = attackers_to_occ(pos, to, occ) & occ;
    let mut stm = pos.stm.flip();
    let mut result = 1i32; // 1 = side-to-move (of the original position) is ok

    loop {
        attackers &= occ;
        let my = attackers & pos.color[stm.index()];
        if my == 0 {
            break;
        }
        // least valuable attacker
        let mut lva_pt = PieceType::King;
        let mut lva_bb = 0u64;
        for pt in PieceType::ALL {
            let b = my & pos.pieces[pt.index()];
            if b != 0 {
                lva_pt = pt;
                lva_bb = b & b.wrapping_neg();
                break;
            }
        }

        occ ^= lva_bb;
        // reveal x-ray attackers
        if matches!(lva_pt, PieceType::Pawn | PieceType::Bishop | PieceType::Queen) {
            attackers |= t.bishop_attacks(to, occ) & bishops;
        }
        if matches!(lva_pt, PieceType::Rook | PieceType::Queen) {
            attackers |= t.rook_attacks(to, occ) & rooks;
        }

        result ^= 1;
        swap = -swap - 1 + val(next);
        next = lva_pt;

        if swap < 0 {
            // capturing with the king into a defended square is illegal - the
            // side that would do that actually loses the exchange
            if lva_pt == PieceType::King && (attackers & occ & pos.color[stm.flip().index()]) != 0 {
                result ^= 1;
            }
            break;
        }
        stm = stm.flip();
    }

    result != 0
}

fn attackers_to_occ(pos: &Position, sq: Square, occ: Bitboard) -> Bitboard {
    let t = tables();
    let mut a = 0u64;
    a |= t.pawn[Color::White.index()][sq as usize] & pos.piece_bb(Color::Black, PieceType::Pawn);
    a |= t.pawn[Color::Black.index()][sq as usize] & pos.piece_bb(Color::White, PieceType::Pawn);
    a |= t.knight[sq as usize] & pos.pieces[PieceType::Knight.index()];
    a |= t.king[sq as usize] & pos.pieces[PieceType::King.index()];
    let bishops = pos.pieces[PieceType::Bishop.index()] | pos.pieces[PieceType::Queen.index()];
    let rooks = pos.pieces[PieceType::Rook.index()] | pos.pieces[PieceType::Queen.index()];
    a |= t.bishop_attacks(sq, occ) & bishops;
    a |= t.rook_attacks(sq, occ) & rooks;
    a
}
