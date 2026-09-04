//! Precomputed attack tables. Leaper attacks are plain arrays; slider attacks
//! use magic bitboards with magics found at startup with a fixed-seed RNG, so
//! the source stays small and the result is deterministic and portable.

use crate::types::*;
use std::sync::OnceLock;

#[inline]
fn shift(b: Bitboard, d: i8) -> Bitboard {
    if d >= 0 {
        b << d
    } else {
        b >> (-d)
    }
}

const NOT_A: Bitboard = !FILE_A;
const NOT_H: Bitboard = !FILE_H;
const NOT_AB: Bitboard = !(FILE_A | (FILE_A << 1));
const NOT_GH: Bitboard = !(FILE_H | (FILE_H >> 1));

pub struct Tables {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    pub pawn: [[Bitboard; 64]; 2], // [color][sq] -> capture targets
    pub between: [[Bitboard; 64]; 64],
    pub line: [[Bitboard; 64]; 64],
    bishop: Magic,
    rook: Magic,
}

struct Magic {
    magics: [u64; 64],
    masks: [Bitboard; 64],
    shifts: [u32; 64],
    offsets: [usize; 64],
    table: Vec<Bitboard>,
}

impl Magic {
    #[inline]
    fn attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq as usize;
        let idx = ((occ & self.masks[s]).wrapping_mul(self.magics[s]) >> self.shifts[s]) as usize;
        self.table[self.offsets[s] + idx]
    }
}

fn slider_attacks(sq: Square, occ: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
    let mut attacks = 0u64;
    let f0 = file_of(sq) as i8;
    let r0 = rank_of(sq) as i8;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as u8;
            attacks |= bb(s);
            if occ & bb(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    attacks
}

const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

fn edge_mask(sq: Square) -> Bitboard {
    let rank_edges = (RANK_1 | RANK_8) & !rank_bb(rank_of(sq));
    let file_edges = (FILE_A | FILE_H) & !file_bb(file_of(sq));
    rank_edges | file_edges
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

fn build_magic(dirs: &[(i8, i8)]) -> Magic {
    let mut magics = [0u64; 64];
    let mut masks = [0u64; 64];
    let mut shifts = [0u32; 64];
    let mut offsets = [0usize; 64];
    let mut table: Vec<Bitboard> = Vec::new();
    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);

    for sq in 0u8..64 {
        let mask = slider_attacks(sq, 0, dirs) & !edge_mask(sq);
        masks[sq as usize] = mask;
        let bits = mask.count_ones();
        shifts[sq as usize] = 64 - bits;
        let size = 1usize << bits;

        // enumerate all blocker subsets of `mask` and their attack sets
        let mut occs = vec![0u64; size];
        let mut atts = vec![0u64; size];
        let mut sub: u64 = 0;
        let mut i = 0;
        loop {
            occs[i] = sub;
            atts[i] = slider_attacks(sq, sub, dirs);
            i += 1;
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }

        // trial magics
        let mut used = vec![0u64; size];
        let mut epoch = vec![0u32; size];
        let mut cur_epoch = 0u32;
        let magic = loop {
            let m = rng.sparse();
            if ((mask.wrapping_mul(m)) >> 56).count_ones() < 6 {
                continue;
            }
            cur_epoch += 1;
            let mut ok = true;
            for k in 0..size {
                let idx = ((occs[k] & mask).wrapping_mul(m) >> (64 - bits)) as usize;
                if epoch[idx] != cur_epoch {
                    epoch[idx] = cur_epoch;
                    used[idx] = atts[k];
                } else if used[idx] != atts[k] {
                    ok = false;
                    break;
                }
            }
            if ok {
                break m;
            }
        };

        magics[sq as usize] = magic;
        offsets[sq as usize] = table.len();
        let mut slot = vec![0u64; size];
        for k in 0..size {
            let idx = ((occs[k] & mask).wrapping_mul(magic) >> (64 - bits)) as usize;
            slot[idx] = atts[k];
        }
        table.extend_from_slice(&slot);
    }

    Magic {
        magics,
        masks,
        shifts,
        offsets,
        table,
    }
}

fn build() -> Tables {
    let mut knight = [0u64; 64];
    let mut king = [0u64; 64];
    let mut pawn = [[0u64; 64]; 2];

    for sq in 0u8..64 {
        let b = bb(sq);
        knight[sq as usize] = shift(b & NOT_H, 17)
            | shift(b & NOT_A, 15)
            | shift(b & NOT_GH, 10)
            | shift(b & NOT_AB, 6)
            | shift(b & NOT_A, -17)
            | shift(b & NOT_H, -15)
            | shift(b & NOT_AB, -10)
            | shift(b & NOT_GH, -6);
        king[sq as usize] = shift(b, 8)
            | shift(b, -8)
            | shift(b & NOT_A, -1)
            | shift(b & NOT_H, 1)
            | shift(b & NOT_A, 7)
            | shift(b & NOT_H, 9)
            | shift(b & NOT_A, -9)
            | shift(b & NOT_H, -7);
        pawn[Color::White.index()][sq as usize] =
            shift(b & NOT_A, 7) | shift(b & NOT_H, 9);
        pawn[Color::Black.index()][sq as usize] =
            shift(b & NOT_H, -7) | shift(b & NOT_A, -9);
    }

    let bishop = build_magic(&BISHOP_DIRS);
    let rook = build_magic(&ROOK_DIRS);

    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];
    for a in 0u8..64 {
        for b2 in 0u8..64 {
            if a == b2 {
                continue;
            }
            for dirs in [&BISHOP_DIRS, &ROOK_DIRS] {
                let aa = slider_attacks(a, 0, dirs);
                if aa & bb(b2) != 0 {
                    let ab = slider_attacks(a, bb(b2), dirs);
                    let ba = slider_attacks(b2, bb(a), dirs);
                    between[a as usize][b2 as usize] = ab & ba;
                    line[a as usize][b2 as usize] =
                        (aa & slider_attacks(b2, 0, dirs)) | bb(a) | bb(b2);
                }
            }
        }
    }

    Tables {
        knight,
        king,
        pawn,
        between,
        line,
        bishop,
        rook,
    }
}

static TABLES: OnceLock<Tables> = OnceLock::new();

pub fn tables() -> &'static Tables {
    TABLES.get_or_init(build)
}

impl Tables {
    #[inline]
    pub fn bishop_attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        self.bishop.attacks(sq, occ)
    }
    #[inline]
    pub fn rook_attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        self.rook.attacks(sq, occ)
    }
    #[inline]
    pub fn queen_attacks(&self, sq: Square, occ: Bitboard) -> Bitboard {
        self.bishop.attacks(sq, occ) | self.rook.attacks(sq, occ)
    }
    #[inline]
    pub fn attacks(&self, pt: PieceType, sq: Square, occ: Bitboard) -> Bitboard {
        match pt {
            PieceType::Knight => self.knight[sq as usize],
            PieceType::King => self.king[sq as usize],
            PieceType::Bishop => self.bishop_attacks(sq, occ),
            PieceType::Rook => self.rook_attacks(sq, occ),
            PieceType::Queen => self.queen_attacks(sq, occ),
            PieceType::Pawn => 0,
        }
    }
}
