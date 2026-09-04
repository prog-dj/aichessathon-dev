//! Legal move generation.

use crate::position::Position;
use crate::tables::tables;
use crate::types::*;
use arrayvec::ArrayVec;

pub type MoveList = ArrayVec<Move, 256>;

const WK_PATH: Bitboard = bb(5) | bb(6); // f1 g1
const WQ_PATH: Bitboard = bb(1) | bb(2) | bb(3); // b1 c1 d1
const BK_PATH: Bitboard = bb(61) | bb(62);
const BQ_PATH: Bitboard = bb(57) | bb(58) | bb(59);

impl Position {
    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.gen(&mut list, false);
        list
    }
    pub fn legal_captures(&self) -> MoveList {
        let mut list = MoveList::new();
        self.gen(&mut list, true);
        list
    }

    fn gen(&self, list: &mut MoveList, captures_only: bool) {
        let t = tables();
        let us = self.stm;
        let them = us.flip();
        let ui = us.index();
        let ti = them.index();
        let occ = self.occupied();
        let own = self.color[ui];
        let enemy = self.color[ti];
        let ksq = self.king_sq(us);

        // checkers
        let checkers = self.attackers_to(ksq, them, occ);
        let num_checkers = checkers.count_ones();

        // squares the enemy attacks with our king removed - king can't step there
        let occ_no_king = occ ^ bb(ksq);
        let danger = self.attack_map(them, occ_no_king);

        // king moves
        let mut kt = t.king[ksq as usize] & !own & !danger;
        if captures_only {
            kt &= enemy;
        }
        push_targets(list, ksq, kt, enemy, self);

        if num_checkers >= 2 {
            return; // double check: only king moves
        }

        // allowed target squares for non-king moves
        let (block_mask, capture_mask) = if num_checkers == 1 {
            let csq = checkers.trailing_zeros() as Square;
            let cpt = self.mailbox[csq as usize].unwrap().1;
            let blocks = if matches!(cpt, PieceType::Bishop | PieceType::Rook | PieceType::Queen) {
                t.between[ksq as usize][csq as usize]
            } else {
                0
            };
            (blocks, checkers)
        } else {
            (!0u64, !0u64)
        };
        let target_mask = block_mask | capture_mask;

        // pinned pieces and their pin rays
        let pinned = self.pinned(ksq, us, occ);

        // pawns
        self.gen_pawns(list, target_mask, pinned, ksq, captures_only);

        // knights
        let mut knights = self.piece_bb(us, PieceType::Knight) & !pinned;
        while knights != 0 {
            let s = knights.trailing_zeros() as Square;
            let mut tt = t.knight[s as usize] & !own & target_mask;
            if captures_only {
                tt &= enemy;
            }
            push_targets(list, s, tt, enemy, self);
            knights &= knights - 1;
        }

        // bishops / rooks / queens
        for pt in [PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let mut b = self.piece_bb(us, pt);
            while b != 0 {
                let s = b.trailing_zeros() as Square;
                b &= b - 1;
                let mut tt = t.attacks(pt, s, occ) & !own & target_mask;
                if captures_only {
                    tt &= enemy;
                }
                if pinned & bb(s) != 0 {
                    tt &= t.line[ksq as usize][s as usize];
                }
                push_targets(list, s, tt, enemy, self);
            }
        }

        // castling (never a capture, skip in captures_only)
        if !captures_only && num_checkers == 0 {
            self.gen_castling(list, danger, occ);
        }
    }

    fn gen_pawns(
        &self,
        list: &mut MoveList,
        target_mask: Bitboard,
        pinned: Bitboard,
        ksq: Square,
        captures_only: bool,
    ) {
        let t = tables();
        let us = self.stm;
        let them = us.flip();
        let occ = self.occupied();
        let enemy = self.color[them.index()];
        let pawns = self.piece_bb(us, PieceType::Pawn);
        let (up, promo_rank, start_rank): (i8, u8, u8) = match us {
            Color::White => (8, 7, 1),
            Color::Black => (-8, 0, 1),
        };
        let start_rank = if us == Color::White { 1 } else { 6 };
        let _ = start_rank;

        let mut b = pawns;
        while b != 0 {
            let from = b.trailing_zeros() as Square;
            b &= b - 1;
            let from_bb = bb(from);
            let is_pinned = pinned & from_bb != 0;
            let pin_line = if is_pinned {
                t.line[ksq as usize][from as usize]
            } else {
                !0u64
            };

            // pushes
            if !captures_only || rank_of(from) == (if us == Color::White { 6 } else { 1 }) {
                let one = (from as i8 + up) as i8;
                if (0..64).contains(&one) {
                    let one = one as Square;
                    if occ & bb(one) == 0 && (pin_line & bb(one) != 0) {
                        if bb(one) & target_mask != 0 {
                            self.add_pawn_move(list, from, one, false);
                        }
                        // double push
                        let sr = if us == Color::White { 1 } else { 6 };
                        if rank_of(from) == sr {
                            let two = (one as i8 + up) as Square;
                            if occ & bb(two) == 0
                                && bb(two) & target_mask != 0
                                && pin_line & bb(two) != 0
                            {
                                list.push(Move::new(from, two, move_flag::DOUBLE_PUSH));
                            }
                        }
                    }
                }
            }

            // captures
            let mut caps = t.pawn[us.index()][from as usize] & enemy & target_mask & pin_line;
            while caps != 0 {
                let to = caps.trailing_zeros() as Square;
                caps &= caps - 1;
                self.add_pawn_move(list, from, to, true);
            }

            // en passant
            if let Some(ep) = self.ep {
                if t.pawn[us.index()][from as usize] & bb(ep) != 0 && pin_line & bb(ep) != 0 {
                    let cap_sq = if us == Color::White { ep - 8 } else { ep + 8 };
                    // legality: remove both pawns, is our king attacked along a rank/diag?
                    let occ2 = (occ ^ from_bb ^ bb(cap_sq)) | bb(ep);
                    let bishops = (self.pieces[PieceType::Bishop.index()]
                        | self.pieces[PieceType::Queen.index()])
                        & enemy;
                    let rooks = (self.pieces[PieceType::Rook.index()]
                        | self.pieces[PieceType::Queen.index()])
                        & enemy;
                    if t.bishop_attacks(ksq, occ2) & bishops == 0
                        && t.rook_attacks(ksq, occ2) & rooks == 0
                    {
                        list.push(Move::new(from, ep, move_flag::EN_PASSANT));
                    }
                }
            }
        }
        let _ = (promo_rank, them);
    }

    #[inline]
    fn add_pawn_move(&self, list: &mut MoveList, from: Square, to: Square, capture: bool) {
        let promo_rank = if self.stm == Color::White { 7 } else { 0 };
        if rank_of(to) == promo_rank {
            let flags = if capture {
                [
                    move_flag::PROMO_QUEEN_CAP,
                    move_flag::PROMO_ROOK_CAP,
                    move_flag::PROMO_BISHOP_CAP,
                    move_flag::PROMO_KNIGHT_CAP,
                ]
            } else {
                [
                    move_flag::PROMO_QUEEN,
                    move_flag::PROMO_ROOK,
                    move_flag::PROMO_BISHOP,
                    move_flag::PROMO_KNIGHT,
                ]
            };
            for f in flags {
                list.push(Move::new(from, to, f));
            }
        } else {
            list.push(Move::new(
                from,
                to,
                if capture {
                    move_flag::CAPTURE
                } else {
                    move_flag::QUIET
                },
            ));
        }
    }

    fn gen_castling(&self, list: &mut MoveList, danger: Bitboard, occ: Bitboard) {
        let us = self.stm;
        match us {
            Color::White => {
                if self.castling & castle::WK != 0
                    && occ & WK_PATH == 0
                    && danger & (bb(4) | WK_PATH) == 0
                {
                    list.push(Move::new(4, 6, move_flag::CASTLE_KING));
                }
                if self.castling & castle::WQ != 0
                    && occ & WQ_PATH == 0
                    && danger & (bb(4) | bb(3) | bb(2)) == 0
                {
                    list.push(Move::new(4, 2, move_flag::CASTLE_QUEEN));
                }
            }
            Color::Black => {
                if self.castling & castle::BK != 0
                    && occ & BK_PATH == 0
                    && danger & (bb(60) | BK_PATH) == 0
                {
                    list.push(Move::new(60, 62, move_flag::CASTLE_KING));
                }
                if self.castling & castle::BQ != 0
                    && occ & BQ_PATH == 0
                    && danger & (bb(60) | bb(59) | bb(58)) == 0
                {
                    list.push(Move::new(60, 58, move_flag::CASTLE_QUEEN));
                }
            }
        }
    }

    /// Pieces of side `c` pinned against their king on `ksq`.
    fn pinned(&self, ksq: Square, c: Color, occ: Bitboard) -> Bitboard {
        let t = tables();
        let them = c.flip();
        let ti = them.index();
        let mut pinned = 0u64;
        let bishops = (self.pieces[PieceType::Bishop.index()]
            | self.pieces[PieceType::Queen.index()])
            & self.color[ti];
        let rooks = (self.pieces[PieceType::Rook.index()] | self.pieces[PieceType::Queen.index()])
            & self.color[ti];
        let mut snipers = (t.bishop_attacks(ksq, 0) & bishops) | (t.rook_attacks(ksq, 0) & rooks);
        while snipers != 0 {
            let s = snipers.trailing_zeros() as Square;
            snipers &= snipers - 1;
            let between = t.between[ksq as usize][s as usize] & occ;
            if between.count_ones() == 1 && between & self.color[c.index()] != 0 {
                pinned |= between;
            }
        }
        pinned
    }

    /// Attackers of colour `by` that hit `sq`.
    pub fn attackers_to(&self, sq: Square, by: Color, occ: Bitboard) -> Bitboard {
        let t = tables();
        let bi = by.index();
        let mut a = 0u64;
        a |= t.pawn[by.flip().index()][sq as usize]
            & self.pieces[PieceType::Pawn.index()]
            & self.color[bi];
        a |= t.knight[sq as usize] & self.pieces[PieceType::Knight.index()] & self.color[bi];
        a |= t.king[sq as usize] & self.pieces[PieceType::King.index()] & self.color[bi];
        let bishops = (self.pieces[PieceType::Bishop.index()]
            | self.pieces[PieceType::Queen.index()])
            & self.color[bi];
        a |= t.bishop_attacks(sq, occ) & bishops;
        let rooks = (self.pieces[PieceType::Rook.index()] | self.pieces[PieceType::Queen.index()])
            & self.color[bi];
        a |= t.rook_attacks(sq, occ) & rooks;
        a
    }

    fn attack_map(&self, by: Color, occ: Bitboard) -> Bitboard {
        let t = tables();
        let bi = by.index();
        let mut a = 0u64;
        let mut pawns = self.pieces[PieceType::Pawn.index()] & self.color[bi];
        while pawns != 0 {
            let s = pawns.trailing_zeros() as Square;
            pawns &= pawns - 1;
            a |= t.pawn[bi][s as usize];
        }
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let mut b = self.pieces[pt.index()] & self.color[bi];
            while b != 0 {
                let s = b.trailing_zeros() as Square;
                b &= b - 1;
                a |= t.attacks(pt, s, occ);
            }
        }
        a
    }
}

#[inline]
fn push_targets(list: &mut MoveList, from: Square, mut tt: Bitboard, enemy: Bitboard, _p: &Position) {
    while tt != 0 {
        let to = tt.trailing_zeros() as Square;
        tt &= tt - 1;
        let flag = if enemy & bb(to) != 0 {
            move_flag::CAPTURE
        } else {
            move_flag::QUIET
        };
        list.push(Move::new(from, to, flag));
    }
}
