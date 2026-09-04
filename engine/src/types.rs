//! Core value types: colours, pieces, squares, moves.

pub type Bitboard = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Piece kinds, indexed 0..6.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn from_index(i: usize) -> PieceType {
        PieceType::ALL[i]
    }
}

/// Board squares, 0 = a1, 7 = h1, 56 = a8, 63 = h8.
pub type Square = u8;

#[inline]
pub const fn square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}
#[inline]
pub const fn file_of(sq: Square) -> u8 {
    sq & 7
}
#[inline]
pub const fn rank_of(sq: Square) -> u8 {
    sq >> 3
}
#[inline]
pub const fn bb(sq: Square) -> Bitboard {
    1u64 << sq
}
#[inline]
pub const fn flip_square(sq: Square) -> Square {
    sq ^ 56
}

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = 0x8080_8080_8080_8080;
pub const RANK_1: Bitboard = 0x0000_0000_0000_00FF;
pub const RANK_2: Bitboard = 0x0000_0000_0000_FF00;
pub const RANK_7: Bitboard = 0x00FF_0000_0000_0000;
pub const RANK_8: Bitboard = 0xFF00_0000_0000_0000;

pub const fn file_bb(file: u8) -> Bitboard {
    FILE_A << file
}
pub const fn rank_bb(rank: u8) -> Bitboard {
    RANK_1 << (8 * rank)
}

/// Move flags packed into the high bits of a u16 move.
pub mod move_flag {
    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const CASTLE_KING: u16 = 2;
    pub const CASTLE_QUEEN: u16 = 3;
    pub const CAPTURE: u16 = 4;
    pub const EN_PASSANT: u16 = 5;
    pub const PROMO_KNIGHT: u16 = 8;
    pub const PROMO_BISHOP: u16 = 9;
    pub const PROMO_ROOK: u16 = 10;
    pub const PROMO_QUEEN: u16 = 11;
    pub const PROMO_KNIGHT_CAP: u16 = 12;
    pub const PROMO_BISHOP_CAP: u16 = 13;
    pub const PROMO_ROOK_CAP: u16 = 14;
    pub const PROMO_QUEEN_CAP: u16 = 15;
}

/// A move: bits 0..6 from, 6..12 to, 12..16 flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move(pub u16);

impl Move {
    #[inline]
    pub fn new(from: Square, to: Square, flag: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flag << 12))
    }
    #[inline]
    pub fn from(self) -> Square {
        (self.0 & 0x3F) as Square
    }
    #[inline]
    pub fn to(self) -> Square {
        ((self.0 >> 6) & 0x3F) as Square
    }
    #[inline]
    pub fn flag(self) -> u16 {
        self.0 >> 12
    }
    #[inline]
    pub fn is_capture(self) -> bool {
        let f = self.flag();
        f == move_flag::CAPTURE || f == move_flag::EN_PASSANT || f >= move_flag::PROMO_KNIGHT_CAP
    }
    #[inline]
    pub fn is_promotion(self) -> bool {
        self.flag() >= move_flag::PROMO_KNIGHT
    }
    #[inline]
    pub fn is_castle(self) -> bool {
        matches!(self.flag(), move_flag::CASTLE_KING | move_flag::CASTLE_QUEEN)
    }
    #[inline]
    pub fn is_en_passant(self) -> bool {
        self.flag() == move_flag::EN_PASSANT
    }
    /// Promotion piece, if any.
    #[inline]
    pub fn promo(self) -> Option<PieceType> {
        match self.flag() {
            move_flag::PROMO_KNIGHT | move_flag::PROMO_KNIGHT_CAP => Some(PieceType::Knight),
            move_flag::PROMO_BISHOP | move_flag::PROMO_BISHOP_CAP => Some(PieceType::Bishop),
            move_flag::PROMO_ROOK | move_flag::PROMO_ROOK_CAP => Some(PieceType::Rook),
            move_flag::PROMO_QUEEN | move_flag::PROMO_QUEEN_CAP => Some(PieceType::Queen),
            _ => None,
        }
    }
    pub fn none() -> Move {
        Move(0)
    }
    pub fn is_none(self) -> bool {
        self.0 == 0
    }

    /// Long algebraic (UCI) form, e.g. `e2e4`, `e7e8q`.
    pub fn to_uci(self) -> String {
        let f = self.from();
        let t = self.to();
        let mut s = String::with_capacity(5);
        s.push((b'a' + file_of(f)) as char);
        s.push((b'1' + rank_of(f)) as char);
        s.push((b'a' + file_of(t)) as char);
        s.push((b'1' + rank_of(t)) as char);
        if let Some(p) = self.promo() {
            s.push(match p {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => unreachable!(),
            });
        }
        s
    }
}

/// Castling rights bitmask.
pub mod castle {
    pub const WK: u8 = 1;
    pub const WQ: u8 = 2;
    pub const BK: u8 = 4;
    pub const BQ: u8 = 8;
    pub const ALL: u8 = 15;
}
