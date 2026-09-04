//! Hand-crafted evaluation. Tapered material + piece-square tables plus a few
//! structural terms. Centipawns, from the side to move's point of view.

use crate::position::Position;
use crate::tables::tables;
use crate::types::*;

pub const MATE: i32 = 30_000;
pub const MATE_IN_MAX: i32 = MATE - 512;

// midgame / endgame piece values
const MG_VAL: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VAL: [i32; 6] = [94, 281, 297, 512, 936, 0];
const PHASE_W: [i32; 6] = [0, 1, 1, 2, 4, 0];
const PHASE_MAX: i32 = 24;

// piece-square tables, White's view, a1..h8 (index 0 = a1). From PeSTO.
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
    // pawn
    [ 0,0,0,0,0,0,0,0, -35,-1,-20,-23,-15,24,38,-22, -26,-4,-4,-10,3,3,33,-12,
      -27,-2,-5,12,17,6,10,-25, -14,13,6,21,23,12,17,-23, -6,7,26,31,65,56,25,-20,
      98,134,61,95,68,126,34,-11, 0,0,0,0,0,0,0,0 ],
    // knight
    [ -105,-21,-58,-33,-17,-28,-19,-23, -29,-53,-12,-3,-1,18,-14,-19, -23,-9,12,10,19,17,25,-16,
      -13,4,16,13,28,19,21,-8, -9,17,19,53,37,69,18,22, -47,60,37,65,84,129,73,44,
      -73,-41,72,36,23,62,7,-17, -167,-89,-34,-49,61,-97,-15,-107 ],
    // bishop
    [ -33,-3,-14,-21,-13,-12,-39,-21, 4,15,16,0,7,21,33,1, 0,15,15,15,14,27,18,10,
      -6,13,13,26,34,12,10,4, -4,5,19,50,37,37,7,-2, -16,37,43,40,35,50,37,-2,
      -26,16,-18,-13,30,59,18,-47, -29,4,-82,-37,-25,-42,7,-8 ],
    // rook
    [ -19,-13,1,17,16,7,-37,-26, -44,-16,-20,-9,-1,11,-6,-71, -45,-25,-16,-17,3,0,-5,-33,
      -36,-26,-12,-1,9,-7,6,-23, -24,-11,7,26,24,35,-8,-20, -5,19,26,36,17,45,61,16,
      27,32,58,62,80,67,26,44, 32,42,32,51,63,9,31,43 ],
    // queen
    [ -1,-18,-9,10,-15,-25,-31,-50, -35,-8,11,2,8,15,-3,1, -14,2,-11,-2,-5,2,14,5,
      -9,-26,-9,-10,-2,-4,3,-3, -27,-27,-16,-16,-1,17,-2,1, -13,-17,7,8,29,56,47,57,
      -24,-39,-5,1,-16,57,28,54, -28,0,29,12,59,44,43,45 ],
    // king
    [ -15,36,12,-54,8,-28,24,14, 1,7,-8,-64,-43,-16,9,8, -14,-14,-22,-46,-44,-30,-15,-27,
      -49,-1,-27,-39,-46,-44,-33,-51, -17,-20,-12,-27,-30,-25,-14,-36, -9,24,2,-16,-20,6,22,-22,
      29,-1,-20,-7,-8,-4,-38,-29, -65,23,16,-15,-56,-34,2,13 ],
];

#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
    [ 0,0,0,0,0,0,0,0, 13,8,8,10,13,0,2,-7, 4,7,-6,1,0,-5,-1,-8, 13,9,-3,-7,-7,-8,3,-1,
      32,24,13,5,-2,4,17,17, 94,100,85,67,56,53,82,84, 178,173,158,134,147,132,165,187, 0,0,0,0,0,0,0,0 ],
    [ -29,-51,-23,-15,-22,-18,-50,-64, -42,-20,-10,-5,-2,-20,-23,-44, -23,-3,-1,15,10,-3,-20,-22,
      -18,-6,16,25,16,17,4,-18, -17,3,22,22,22,11,8,-18, -24,-20,10,9,-1,-9,-19,-41,
      -25,-8,-25,-2,-9,-25,-24,-52, -58,-38,-13,-28,-31,-27,-63,-99 ],
    [ -23,-9,-23,-5,-9,-16,-5,-17, -14,-18,-7,-1,4,-9,-15,-27, -12,-3,8,10,13,3,-7,-15,
      -6,3,13,19,7,10,-3,-9, -3,9,12,9,14,10,3,2, 2,-8,0,-1,-2,6,0,4,
      -8,-4,7,-12,-3,-13,-4,-14, -14,-21,-11,-8,-7,-9,-17,-24 ],
    [ -9,2,3,-1,-5,-13,4,-20, -6,-6,0,2,-9,-9,-11,-3, -4,0,-5,-1,-7,-12,-8,-16,
      3,5,8,4,-5,-6,-8,-11, 4,3,13,1,2,1,-1,2, 7,7,7,5,4,-3,-5,-3,
      11,13,13,11,-3,3,8,3, 13,10,18,15,12,12,8,5 ],
    [ -33,-28,-22,-43,-5,-32,-20,-41, -22,-23,-30,-16,-16,-23,-36,-32, -16,-27,15,6,9,17,10,5,
      -18,28,19,47,31,34,39,23, 3,22,24,45,57,40,57,36, -20,6,9,49,47,35,19,9,
      -17,20,32,41,58,25,30,0, -9,22,22,27,27,19,10,20 ],
    [ -53,-34,-21,-11,-28,-14,-24,-43, -27,-11,4,13,14,4,-5,-17, -19,-3,11,21,23,16,7,-9,
      -18,-4,21,24,27,23,9,-11, -8,22,24,27,26,33,26,3, 10,17,23,15,20,45,44,13,
      -12,17,14,17,17,38,23,11, -74,-35,-18,-18,-11,15,4,-17 ],
];

fn phase(pos: &Position) -> i32 {
    let mut p = 0;
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        p += PHASE_W[pt.index()] * pos.pieces[pt.index()].count_ones() as i32;
    }
    p.min(PHASE_MAX)
}

#[inline]
fn pst_sq(sq: Square, c: Color) -> usize {
    (if c == Color::White { sq } else { flip_square(sq) }) as usize
}

const PASSED_MG: [i32; 8] = [0, 5, 10, 15, 30, 55, 90, 0];
const PASSED_EG: [i32; 8] = [0, 10, 18, 30, 55, 95, 160, 0];
const ISOLATED: i32 = 12;
const DOUBLED: i32 = 10;
const BISHOP_PAIR: i32 = 25;
const ROOK_OPEN: i32 = 22;
const ROOK_HALF: i32 = 10;
const TEMPO: i32 = 14;

pub fn evaluate(pos: &Position) -> i32 {
    let t = tables();
    let ph = phase(pos);
    let occ = pos.occupied();
    let mut mg = 0i32;
    let mut eg = 0i32;

    let mut safety = [0i32; 2]; // enemy pressure on [white king, black king]
    let mut mobility_mg = [0i32; 2];

    for c in [Color::White, Color::Black] {
        let sign = if c == Color::White { 1 } else { -1 };
        let ci = c.index();
        let them = c.flip();
        let king_ring = t.king[pos.king_sq(them) as usize] | bb(pos.king_sq(them));

        for pt in PieceType::ALL {
            let mut b = pos.piece_bb(c, pt);
            while b != 0 {
                let sq = b.trailing_zeros() as Square;
                b &= b - 1;
                let idx = pst_sq(sq, c);
                mg += sign * (MG_VAL[pt.index()] + MG_PST[pt.index()][idx]);
                eg += sign * (EG_VAL[pt.index()] + EG_PST[pt.index()][idx]);

                match pt {
                    PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
                        let att = t.attacks(pt, sq, occ) & !pos.color[ci];
                        mobility_mg[ci] += (att.count_ones() as i32) * 2;
                        let ring_hits = (att & king_ring).count_ones() as i32;
                        if ring_hits > 0 {
                            safety[them.index()] += ring_hits
                                * match pt {
                                    PieceType::Knight | PieceType::Bishop => 2,
                                    PieceType::Rook => 3,
                                    _ => 5,
                                };
                        }
                    }
                    _ => {}
                }
            }
        }

        // bishop pair
        if pos.piece_bb(c, PieceType::Bishop).count_ones() >= 2 {
            mg += sign * BISHOP_PAIR;
            eg += sign * BISHOP_PAIR;
        }

        // pawns: passed / isolated / doubled
        let own_pawns = pos.piece_bb(c, PieceType::Pawn);
        let enemy_pawns = pos.piece_bb(them, PieceType::Pawn);
        let mut p = own_pawns;
        while p != 0 {
            let sq = p.trailing_zeros() as Square;
            p &= p - 1;
            let f = file_of(sq);
            let file_mask = file_bb(f);
            let adj = (if f > 0 { file_bb(f - 1) } else { 0 })
                | (if f < 7 { file_bb(f + 1) } else { 0 });
            // isolated
            if own_pawns & adj == 0 {
                mg -= sign * ISOLATED;
                eg -= sign * ISOLATED;
            }
            // doubled
            if (own_pawns & file_mask).count_ones() > 1 {
                mg -= sign * DOUBLED / 2;
                eg -= sign * DOUBLED;
            }
            // passed
            let front = if c == Color::White {
                pawn_front_span_white(sq)
            } else {
                pawn_front_span_black(sq)
            };
            if (file_mask | adj) & front & enemy_pawns == 0 {
                let rr = if c == Color::White {
                    rank_of(sq)
                } else {
                    7 - rank_of(sq)
                } as usize;
                mg += sign * PASSED_MG[rr];
                eg += sign * PASSED_EG[rr];
            }
        }

        // rooks on open / half-open files
        let mut r = pos.piece_bb(c, PieceType::Rook);
        while r != 0 {
            let sq = r.trailing_zeros() as Square;
            r &= r - 1;
            let fm = file_bb(file_of(sq));
            if fm & pos.pieces[PieceType::Pawn.index()] == 0 {
                mg += sign * ROOK_OPEN;
            } else if fm & own_pawns == 0 {
                mg += sign * ROOK_HALF;
            }
        }
    }

    // king danger -> superlinear penalty
    let wd = king_danger(safety[Color::White.index()]);
    let bd = king_danger(safety[Color::Black.index()]);
    mg += bd - wd;

    mg += mobility_mg[Color::White.index()] - mobility_mg[Color::Black.index()];

    let score = (mg * ph + eg * (PHASE_MAX - ph)) / PHASE_MAX;
    let stm = if pos.stm == Color::White { score } else { -score };
    stm + TEMPO
}

#[inline]
fn king_danger(units: i32) -> i32 {
    let u = units.min(40);
    (u * u * 3) / 8
}

fn pawn_front_span_white(sq: Square) -> Bitboard {
    let r = rank_of(sq);
    let mut m = 0u64;
    for rr in (r + 1)..8 {
        m |= rank_bb(rr);
    }
    m
}
fn pawn_front_span_black(sq: Square) -> Bitboard {
    let r = rank_of(sq);
    let mut m = 0u64;
    for rr in 0..r {
        m |= rank_bb(rr);
    }
    m
}
