//! Transposition table: fixed-size, always-replace-by-depth with aging.

use crate::types::Move;

#[derive(Clone, Copy, PartialEq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    key: u32,
    mv: u16,
    score: i16,
    depth: i8,
    bound: u8, // 0 none, 1 exact, 2 lower, 3 upper
    age: u8,
}

impl Entry {
    const EMPTY: Entry = Entry {
        key: 0,
        mv: 0,
        score: 0,
        depth: -1,
        bound: 0,
        age: 0,
    };
}

pub struct Tt {
    table: Vec<Entry>,
    mask: usize,
    age: u8,
}

pub struct Probe {
    pub mv: Move,
    pub score: i32,
    pub depth: i8,
    pub bound: Bound,
}

impl Tt {
    pub fn new(mb: usize) -> Tt {
        let bytes = mb * 1024 * 1024;
        let n = (bytes / std::mem::size_of::<Entry>()).next_power_of_two() / 2;
        let n = n.max(1024);
        Tt {
            table: vec![Entry::EMPTY; n],
            mask: n - 1,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().for_each(|e| *e = Entry::EMPTY);
        self.age = 0;
    }

    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline]
    fn idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    pub fn probe(&self, key: u64) -> Option<Probe> {
        let e = self.table[self.idx(key)];
        if e.bound == 0 || e.key != (key >> 32) as u32 {
            return None;
        }
        Some(Probe {
            mv: Move(e.mv),
            score: e.score as i32,
            depth: e.depth,
            bound: match e.bound {
                1 => Bound::Exact,
                2 => Bound::Lower,
                _ => Bound::Upper,
            },
        })
    }

    pub fn store(&mut self, key: u64, mv: Move, score: i32, depth: i8, bound: Bound) {
        let i = self.idx(key);
        let e = &mut self.table[i];
        let k32 = (key >> 32) as u32;
        // replace if empty, same position, older search, or shallower
        let replace = e.bound == 0
            || e.key == k32
            || e.age != self.age
            || (depth as i16) >= e.depth as i16 - 2;
        if !replace {
            return;
        }
        let mv = if mv.is_none() && e.key == k32 {
            e.mv
        } else {
            mv.0
        };
        *e = Entry {
            key: k32,
            mv,
            score: score.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            depth,
            bound: match bound {
                Bound::Exact => 1,
                Bound::Lower => 2,
                Bound::Upper => 3,
            },
            age: self.age,
        };
    }
}
