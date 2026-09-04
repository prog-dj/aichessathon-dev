//! Board state: piece bitboards, make / unmake, FEN, Zobrist.

use crate::tables::tables;
use crate::types::*;

#[derive(Clone)]
pub struct Position {
    /// piece bitboards, indexed [piece_type]
    pub pieces: [Bitboard; 6],
    /// occupancy per colour, indexed [colour]
    pub color: [Bitboard; 2],
    pub mailbox: [Option<(Color, PieceType)>; 64],
    pub stm: Color,
    pub castling: u8,
    pub ep: Option<Square>,
    pub halfmove: u16,
    pub fullmove: u16,
    pub key: u64,
    history: Vec<Undo>,
}

#[derive(Clone)]
struct Undo {
    mv: Move,
    captured: Option<PieceType>,
    castling: u8,
    ep: Option<Square>,
    halfmove: u16,
    key: u64,
}

// ---- Zobrist keys -------------------------------------------------------------

struct Zobrist {
    psq: [[[u64; 64]; 6]; 2],
    castling: [u64; 16],
    ep_file: [u64; 8],
    stm: u64,
}

fn zobrist() -> &'static Zobrist {
    use std::sync::OnceLock;
    static Z: OnceLock<Zobrist> = OnceLock::new();
    Z.get_or_init(|| {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut psq = [[[0u64; 64]; 6]; 2];
        for c in 0..2 {
            for p in 0..6 {
                for sq in 0..64 {
                    psq[c][p][sq] = next();
                }
            }
        }
        let mut castling = [0u64; 16];
        for c in castling.iter_mut() {
            *c = next();
        }
        let mut ep_file = [0u64; 8];
        for e in ep_file.iter_mut() {
            *e = next();
        }
        let stm = next();
        Zobrist {
            psq,
            castling,
            ep_file,
            stm,
        }
    })
}

impl Position {
    pub fn empty() -> Position {
        Position {
            pieces: [0; 6],
            color: [0; 2],
            mailbox: [None; 64],
            stm: Color::White,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            key: 0,
            history: Vec::with_capacity(64),
        }
    }

    pub fn startpos() -> Position {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    #[inline]
    pub fn occupied(&self) -> Bitboard {
        self.color[0] | self.color[1]
    }
    #[inline]
    pub fn our(&self, c: Color) -> Bitboard {
        self.color[c.index()]
    }
    #[inline]
    pub fn piece_bb(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.index()] & self.color[c.index()]
    }
    #[inline]
    pub fn king_sq(&self, c: Color) -> Square {
        (self.pieces[PieceType::King.index()] & self.color[c.index()]).trailing_zeros() as Square
    }

    #[inline]
    fn put(&mut self, c: Color, pt: PieceType, sq: Square) {
        self.pieces[pt.index()] |= bb(sq);
        self.color[c.index()] |= bb(sq);
        self.mailbox[sq as usize] = Some((c, pt));
        self.key ^= zobrist().psq[c.index()][pt.index()][sq as usize];
    }
    #[inline]
    fn remove(&mut self, c: Color, pt: PieceType, sq: Square) {
        self.pieces[pt.index()] &= !bb(sq);
        self.color[c.index()] &= !bb(sq);
        self.mailbox[sq as usize] = None;
        self.key ^= zobrist().psq[c.index()][pt.index()][sq as usize];
    }
    #[inline]
    fn move_piece(&mut self, c: Color, pt: PieceType, from: Square, to: Square) {
        self.remove(c, pt, from);
        self.put(c, pt, to);
    }

    // ---- FEN ---------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut pos = Position::empty();
        let mut parts = fen.split_whitespace();
        let board = parts.next().ok_or("empty fen")?;
        let mut rank = 7i32;
        let mut file = 0i32;
        for ch in board.chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='9' => file += ch.to_digit(10).unwrap() as i32,
                _ => {
                    let c = if ch.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let pt = match ch.to_ascii_lowercase() {
                        'p' => PieceType::Pawn,
                        'n' => PieceType::Knight,
                        'b' => PieceType::Bishop,
                        'r' => PieceType::Rook,
                        'q' => PieceType::Queen,
                        'k' => PieceType::King,
                        _ => return Err(format!("bad piece '{ch}'")),
                    };
                    if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                        return Err("fen board overflow".into());
                    }
                    pos.put(c, pt, square(file as u8, rank as u8));
                    file += 1;
                }
            }
        }
        pos.stm = match parts.next() {
            Some("w") => Color::White,
            Some("b") => Color::Black,
            _ => return Err("bad side to move".into()),
        };
        if pos.stm == Color::Black {
            pos.key ^= zobrist().stm;
        }
        let rights = parts.next().unwrap_or("-");
        for ch in rights.chars() {
            match ch {
                'K' => pos.castling |= castle::WK,
                'Q' => pos.castling |= castle::WQ,
                'k' => pos.castling |= castle::BK,
                'q' => pos.castling |= castle::BQ,
                '-' => {}
                _ => return Err("bad castling".into()),
            }
        }
        pos.key ^= zobrist().castling[pos.castling as usize];
        pos.ep = match parts.next() {
            Some("-") | None => None,
            Some(sq) => {
                let b = sq.as_bytes();
                if b.len() != 2 {
                    return Err("bad ep".into());
                }
                let s = square(b[0] - b'a', b[1] - b'1');
                Some(s)
            }
        };
        if let Some(s) = pos.ep {
            pos.key ^= zobrist().ep_file[file_of(s) as usize];
        }
        pos.halfmove = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        pos.fullmove = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.mailbox[square(file, rank) as usize] {
                    None => empty += 1,
                    Some((c, pt)) => {
                        if empty > 0 {
                            s.push((b'0' + empty) as char);
                            empty = 0;
                        }
                        let ch = match pt {
                            PieceType::Pawn => 'p',
                            PieceType::Knight => 'n',
                            PieceType::Bishop => 'b',
                            PieceType::Rook => 'r',
                            PieceType::Queen => 'q',
                            PieceType::King => 'k',
                        };
                        s.push(if c == Color::White {
                            ch.to_ascii_uppercase()
                        } else {
                            ch
                        });
                    }
                }
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bitv, ch) in [
                (castle::WK, 'K'),
                (castle::WQ, 'Q'),
                (castle::BK, 'k'),
                (castle::BQ, 'q'),
            ] {
                if self.castling & bitv != 0 {
                    s.push(ch);
                }
            }
        }
        s.push(' ');
        match self.ep {
            None => s.push('-'),
            Some(sq) => {
                s.push((b'a' + file_of(sq)) as char);
                s.push((b'1' + rank_of(sq)) as char);
            }
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    // ---- attack queries --------------------------------------------------

    /// Is `sq` attacked by side `by`?
    pub fn attacked_by(&self, sq: Square, by: Color, occ: Bitboard) -> bool {
        let t = tables();
        let bi = by.index();
        if t.pawn[by.flip().index()][sq as usize] & self.pieces[PieceType::Pawn.index()] & self.color[bi]
            != 0
        {
            return true;
        }
        if t.knight[sq as usize] & self.pieces[PieceType::Knight.index()] & self.color[bi] != 0 {
            return true;
        }
        if t.king[sq as usize] & self.pieces[PieceType::King.index()] & self.color[bi] != 0 {
            return true;
        }
        let bishops = (self.pieces[PieceType::Bishop.index()]
            | self.pieces[PieceType::Queen.index()])
            & self.color[bi];
        if t.bishop_attacks(sq, occ) & bishops != 0 {
            return true;
        }
        let rooks = (self.pieces[PieceType::Rook.index()] | self.pieces[PieceType::Queen.index()])
            & self.color[bi];
        if t.rook_attacks(sq, occ) & rooks != 0 {
            return true;
        }
        false
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.attacked_by(self.king_sq(self.stm), self.stm.flip(), self.occupied())
    }

    /// All squares attacked by `by` (used for king-danger and eval).
    pub fn attacks_by(&self, by: Color) -> Bitboard {
        let t = tables();
        let bi = by.index();
        let occ = self.occupied();
        let mut a = 0u64;
        let mut pawns = self.pieces[PieceType::Pawn.index()] & self.color[bi];
        while pawns != 0 {
            let s = pawns.trailing_zeros() as Square;
            a |= t.pawn[bi][s as usize];
            pawns &= pawns - 1;
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
                a |= t.attacks(pt, s, occ);
                b &= b - 1;
            }
        }
        a
    }

    // ---- make / unmake --------------------------------------------------

    fn set_castling(&mut self, new: u8) {
        if new != self.castling {
            self.key ^= zobrist().castling[self.castling as usize];
            self.key ^= zobrist().castling[new as usize];
            self.castling = new;
        }
    }
    fn set_ep(&mut self, new: Option<Square>) {
        if let Some(s) = self.ep {
            self.key ^= zobrist().ep_file[file_of(s) as usize];
        }
        if let Some(s) = new {
            self.key ^= zobrist().ep_file[file_of(s) as usize];
        }
        self.ep = new;
    }

    pub fn make_move(&mut self, mv: Move) {
        let us = self.stm;
        let them = us.flip();
        let from = mv.from();
        let to = mv.to();
        let (_, moving) = self.mailbox[from as usize].expect("move from empty square");

        let mut captured = None;
        self.history.push(Undo {
            mv,
            captured: None, // patched below
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            key: self.key,
        });

        let old_ep = self.ep;
        self.set_ep(None);
        self.halfmove += 1;

        if mv.is_en_passant() {
            let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
            self.remove(them, PieceType::Pawn, cap_sq);
            self.move_piece(us, PieceType::Pawn, from, to);
            captured = Some(PieceType::Pawn);
            self.halfmove = 0;
        } else if mv.is_castle() {
            self.move_piece(us, PieceType::King, from, to);
            let (rf, rt) = match mv.flag() {
                move_flag::CASTLE_KING => (to + 1, to - 1),
                _ => (to - 2, to + 1),
            };
            self.move_piece(us, PieceType::Rook, rf, rt);
        } else {
            if let Some((_, cpt)) = self.mailbox[to as usize] {
                self.remove(them, cpt, to);
                captured = Some(cpt);
                self.halfmove = 0;
            }
            if let Some(promo) = mv.promo() {
                self.remove(us, PieceType::Pawn, from);
                self.put(us, promo, to);
                self.halfmove = 0;
            } else {
                self.move_piece(us, moving, from, to);
            }
            if moving == PieceType::Pawn {
                self.halfmove = 0;
                if mv.flag() == move_flag::DOUBLE_PUSH {
                    let ep = if us == Color::White { from + 8 } else { from - 8 };
                    // only set ep if an enemy pawn can actually take (matches FEN norms loosely)
                    self.set_ep(Some(ep));
                }
            }
        }
        let _ = old_ep;

        // castling rights update
        let mut rights = self.castling;
        for sq in [from, to] {
            rights &= match sq {
                4 => !(castle::WK | castle::WQ),
                0 => !castle::WQ,
                7 => !castle::WK,
                60 => !(castle::BK | castle::BQ),
                56 => !castle::BQ,
                63 => !castle::BK,
                _ => castle::ALL,
            };
        }
        self.set_castling(rights);

        if us == Color::Black {
            self.fullmove += 1;
        }
        self.stm = them;
        self.key ^= zobrist().stm;

        let last = self.history.last_mut().unwrap();
        last.captured = captured;
    }

    pub fn unmake_move(&mut self) {
        let undo = self.history.pop().expect("unmake with empty history");
        let mv = undo.mv;
        let them = self.stm;
        let us = them.flip();
        let from = mv.from();
        let to = mv.to();

        self.stm = us;
        if us == Color::Black {
            self.fullmove -= 1;
        }

        if mv.is_en_passant() {
            self.move_piece(us, PieceType::Pawn, to, from);
            let cap_sq = if us == Color::White { to - 8 } else { to + 8 };
            self.put(them, PieceType::Pawn, cap_sq);
        } else if mv.is_castle() {
            self.move_piece(us, PieceType::King, to, from);
            let (rf, rt) = match mv.flag() {
                move_flag::CASTLE_KING => (to + 1, to - 1),
                _ => (to - 2, to + 1),
            };
            self.move_piece(us, PieceType::Rook, rt, rf);
        } else if let Some(promo) = mv.promo() {
            self.remove(us, promo, to);
            self.put(us, PieceType::Pawn, from);
            if let Some(cpt) = undo.captured {
                self.put(them, cpt, to);
            }
        } else {
            let (_, moving) = self.mailbox[to as usize].expect("unmake to empty");
            self.move_piece(us, moving, to, from);
            if let Some(cpt) = undo.captured {
                self.put(them, cpt, to);
            }
        }

        self.castling = undo.castling;
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
    }

    /// Null move (side to move passes). Only legal when not in check.
    pub fn make_null(&mut self) {
        self.history.push(Undo {
            mv: Move::none(),
            captured: None,
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            key: self.key,
        });
        self.set_ep(None);
        self.stm = self.stm.flip();
        self.key ^= zobrist().stm;
        self.halfmove += 1;
    }
    pub fn unmake_null(&mut self) {
        let undo = self.history.pop().unwrap();
        self.stm = self.stm.flip();
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
    }

    /// Repetition / 50-move / insufficient-material draw, as seen from a search.
    /// One repetition in the window is enough - a position we can reach twice we
    /// can reach a third time and the opponent claims.
    pub fn is_draw(&self, _ply: i32) -> bool {
        if self.halfmove >= 100 {
            return true;
        }
        let n = self.history.len();
        let limit = (self.halfmove as usize).min(n);
        let mut back = 4;
        while back <= limit {
            if self.history[n - back].key == self.key {
                return true;
            }
            back += 2;
        }
        self.insufficient_material()
    }

    pub fn insufficient_material(&self) -> bool {
        let p = &self.pieces;
        if p[PieceType::Pawn.index()] != 0
            || p[PieceType::Rook.index()] != 0
            || p[PieceType::Queen.index()] != 0
        {
            return false;
        }
        let minors = (p[PieceType::Knight.index()] | p[PieceType::Bishop.index()]).count_ones();
        minors <= 1
    }

    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        (self.color[c.index()]
            & !(self.pieces[PieceType::Pawn.index()] | self.pieces[PieceType::King.index()]))
            != 0
    }
}
