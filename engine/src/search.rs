//! Iterative-deepening alpha-beta: PVS, TT, killers, history, null move, LMR,
//! reverse futility, quiescence with SEE.

use crate::eval::{evaluate, MATE, MATE_IN_MAX};
use crate::movegen::MoveList;
use crate::position::Position;
use crate::see::see_ge;
use crate::tt::{Bound, Tt};
use crate::types::*;
use std::time::Instant;

const MAX_PLY: usize = 128;
const INF: i32 = MATE + 1;

pub struct Limits {
    pub soft_ms: u64,
    pub hard_ms: u64,
    pub max_depth: i32,
    pub max_nodes: u64,
}

pub struct Searcher {
    pub tt: Tt,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    nodes: u64,
    start: Instant,
    hard_ms: u64,
    stop: bool,
    root_ply: i32,
    seldepth: i32,
}

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

impl Searcher {
    pub fn new(tt_mb: usize) -> Searcher {
        Searcher {
            tt: Tt::new(tt_mb),
            killers: [[Move::none(); 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            nodes: 0,
            start: Instant::now(),
            hard_ms: 0,
            stop: false,
            root_ply: 0,
            seldepth: 0,
        }
    }

    pub fn search(&mut self, pos: &mut Position, limits: &Limits) -> SearchResult {
        self.nodes = 0;
        self.start = Instant::now();
        self.hard_ms = limits.hard_ms;
        self.stop = false;
        self.root_ply = pos.fullmove as i32 * 2;
        self.killers = [[Move::none(); 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.tt.new_search();

        let mut best = Move::none();
        let mut score = 0;
        let root_moves = pos.legal_moves();
        if root_moves.is_empty() {
            return SearchResult {
                best: Move::none(),
                score: 0,
                depth: 0,
                nodes: 0,
            };
        }
        best = root_moves[0];

        let mut depth = 1;
        while depth <= limits.max_depth {
            self.seldepth = 0;
            let mut alpha = -INF;
            let mut beta = INF;
            let mut window = 25;
            if depth >= 4 {
                alpha = (score - window).max(-INF);
                beta = (score + window).min(INF);
            }

            let s = loop {
                let s = self.negamax(pos, depth, 0, alpha, beta, true);
                if self.stop {
                    break s;
                }
                if s <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (s - window).max(-INF);
                    window *= 2;
                } else if s >= beta {
                    beta = (s + window).min(INF);
                    window *= 2;
                } else {
                    break s;
                }
            };

            if self.stop && depth > 1 {
                break;
            }
            score = s;
            if let Some(p) = self.tt.probe(pos.key) {
                if !p.mv.is_none() {
                    best = p.mv;
                }
            }

            // time: after finishing a depth, decide whether to start another
            let elapsed = self.start.elapsed().as_millis() as u64;
            if elapsed >= limits.soft_ms || self.nodes >= limits.max_nodes {
                break;
            }
            depth += 1;
        }

        SearchResult {
            best,
            score,
            depth: depth.min(limits.max_depth),
            nodes: self.nodes,
        }
    }

    #[inline]
    fn check_time(&mut self) {
        if self.nodes & 2047 == 0
            && self.start.elapsed().as_millis() as u64 >= self.hard_ms
        {
            self.stop = true;
        }
    }

    fn negamax(
        &mut self,
        pos: &mut Position,
        mut depth: i32,
        ply: i32,
        mut alpha: i32,
        mut beta: i32,
        _is_pv_hint: bool,
    ) -> i32 {
        if self.stop {
            return 0;
        }
        self.nodes += 1;
        self.check_time();
        let is_pv = beta - alpha > 1;
        if ply as usize > self.seldepth as usize {
            self.seldepth = ply;
        }

        if ply > 0 && pos.is_draw(ply) {
            return 0;
        }
        if ply >= MAX_PLY as i32 - 1 {
            return evaluate(pos);
        }

        // mate distance pruning
        alpha = alpha.max(-MATE + ply);
        beta = beta.min(MATE - ply - 1);
        if alpha >= beta {
            return alpha;
        }

        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }

        if depth <= 0 {
            return self.quiescence(pos, ply, alpha, beta);
        }

        // TT probe
        let mut tt_move = Move::none();
        if let Some(p) = self.tt.probe(pos.key) {
            tt_move = p.mv;
            if !is_pv && p.depth as i32 >= depth {
                let s = tt_score_from(p.score, ply);
                match p.bound {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let static_eval = if in_check { -INF } else { evaluate(pos) };

        // reverse futility
        if !is_pv
            && !in_check
            && depth <= 6
            && static_eval - 80 * depth >= beta
            && beta.abs() < MATE_IN_MAX
        {
            return static_eval;
        }

        // null move
        if !is_pv
            && !in_check
            && depth >= 3
            && static_eval >= beta
            && pos.has_non_pawn_material(pos.stm)
        {
            let r = 3 + depth / 4;
            pos.make_null();
            let s = -self.negamax(pos, depth - r, ply + 1, -beta, -beta + 1, false);
            pos.unmake_null();
            if s >= beta && s.abs() < MATE_IN_MAX {
                return beta;
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if in_check { -MATE + ply } else { 0 };
        }
        self.order_moves(pos, &mut moves, tt_move, ply);

        let old_alpha = alpha;
        let mut best = -INF;
        let mut best_move = Move::none();
        let mut quiets_tried: ArrayVec8 = ArrayVec8::new();

        for (i, &mv) in moves.iter().enumerate() {
            let quiet = !mv.is_capture() && !mv.is_promotion();

            // late move pruning
            if !is_pv
                && !in_check
                && quiet
                && depth <= 4
                && i as i32 >= 4 + depth * depth
                && best > -MATE_IN_MAX
            {
                continue;
            }
            // futility on quiets near the leaf
            if !is_pv
                && !in_check
                && quiet
                && depth <= 3
                && static_eval + 90 * depth <= alpha
                && best > -MATE_IN_MAX
                && i > 0
            {
                continue;
            }

            pos.make_move(mv);
            let gives_check = pos.in_check();
            let mut new_depth = depth - 1;

            let score;
            if i == 0 {
                score = -self.negamax(pos, new_depth, ply + 1, -beta, -alpha, is_pv);
            } else {
                // late move reduction
                let mut r = 0;
                if depth >= 3 && quiet && !gives_check && !in_check {
                    r = 1 + (i >= 6) as i32 + (depth >= 6 && i >= 12) as i32;
                    if is_pv {
                        r -= 1;
                    }
                    r = r.clamp(0, new_depth - 1);
                }
                let mut s = -self.negamax(pos, new_depth - r, ply + 1, -alpha - 1, -alpha, false);
                if s > alpha && r > 0 {
                    s = -self.negamax(pos, new_depth, ply + 1, -alpha - 1, -alpha, false);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(pos, new_depth, ply + 1, -beta, -alpha, true);
                }
                score = s;
            }
            let _ = &mut new_depth;
            pos.unmake_move();

            if self.stop {
                return 0;
            }

            if score > best {
                best = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        if quiet {
                            self.store_killer(ply, mv);
                            self.bump_history(pos.stm, mv, depth);
                            for &q in quiets_tried.iter() {
                                self.drop_history(pos.stm, q, depth);
                            }
                        }
                        break;
                    }
                }
            }
            if quiet {
                let _ = quiets_tried.try_push(mv); // full is fine: just fewer history maluses
            }
        }

        let bound = if best <= old_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt
            .store(pos.key, best_move, tt_score_to(best, ply), depth as i8, bound);
        best
    }

    fn quiescence(&mut self, pos: &mut Position, ply: i32, mut alpha: i32, beta: i32) -> i32 {
        if self.stop {
            return 0;
        }
        self.nodes += 1;
        self.check_time();
        if ply >= MAX_PLY as i32 - 1 {
            return evaluate(pos);
        }

        let in_check = pos.in_check();
        let stand_pat;
        if in_check {
            stand_pat = -INF;
        } else {
            stand_pat = evaluate(pos);
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves = if in_check {
            pos.legal_moves()
        } else {
            pos.legal_captures()
        };
        if moves.is_empty() {
            return if in_check { -MATE + ply } else { stand_pat };
        }
        self.order_moves(pos, &mut moves, Move::none(), ply);

        let mut best = stand_pat;
        for &mv in moves.iter() {
            if !in_check {
                // delta pruning + losing-capture skip
                if !mv.is_promotion() && !see_ge(pos, mv, -20) {
                    continue;
                }
            }
            pos.make_move(mv);
            let s = -self.quiescence(pos, ply + 1, -beta, -alpha);
            pos.unmake_move();
            if self.stop {
                return 0;
            }
            if s > best {
                best = s;
                if s > alpha {
                    alpha = s;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    fn order_moves(&self, pos: &Position, moves: &mut MoveList, tt_move: Move, ply: i32) {
        let stm = pos.stm.index();
        let killers = if (ply as usize) < MAX_PLY {
            self.killers[ply as usize]
        } else {
            [Move::none(); 2]
        };
        let mut scored: ArrayVec<(i32, Move), 256> = ArrayVec::new();
        for &mv in moves.iter() {
            let s = if mv == tt_move {
                1_000_000
            } else if mv.is_capture() {
                let victim = pos
                    .mailbox
                    .get(mv.to() as usize)
                    .and_then(|o| *o)
                    .map(|(_, pt)| pt.index() as i32)
                    .unwrap_or(0);
                let attacker = pos.mailbox[mv.from() as usize]
                    .map(|(_, pt)| pt.index() as i32)
                    .unwrap_or(0);
                100_000 + victim * 16 - attacker
            } else if mv == killers[0] {
                90_000
            } else if mv == killers[1] {
                80_000
            } else {
                self.history[stm][mv.from() as usize][mv.to() as usize]
            };
            scored.push((s, mv));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        moves.clear();
        for (_, mv) in scored {
            moves.push(mv);
        }
    }

    fn store_killer(&mut self, ply: i32, mv: Move) {
        if (ply as usize) >= MAX_PLY {
            return;
        }
        let k = &mut self.killers[ply as usize];
        if k[0] != mv {
            k[1] = k[0];
            k[0] = mv;
        }
    }
    fn bump_history(&mut self, c: Color, mv: Move, depth: i32) {
        let e = &mut self.history[c.index()][mv.from() as usize][mv.to() as usize];
        *e += depth * depth;
        if *e > 1 << 20 {
            for a in self.history.iter_mut() {
                for b in a.iter_mut() {
                    for v in b.iter_mut() {
                        *v /= 2;
                    }
                }
            }
        }
    }
    fn drop_history(&mut self, c: Color, mv: Move, depth: i32) {
        let e = &mut self.history[c.index()][mv.from() as usize][mv.to() as usize];
        *e -= depth * depth;
    }
}

use arrayvec::ArrayVec;
type ArrayVec8 = ArrayVec<Move, 96>;

#[inline]
fn tt_score_to(s: i32, ply: i32) -> i32 {
    if s >= MATE_IN_MAX {
        s + ply
    } else if s <= -MATE_IN_MAX {
        s - ply
    } else {
        s
    }
}
#[inline]
fn tt_score_from(s: i32, ply: i32) -> i32 {
    if s >= MATE_IN_MAX {
        s - ply
    } else if s <= -MATE_IN_MAX {
        s + ply
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best(fen: &str, depth: i32) -> (String, i32) {
        let mut pos = Position::from_fen(fen).unwrap();
        let mut s = Searcher::new(16);
        let limits = Limits {
            soft_ms: u64::MAX,
            hard_ms: u64::MAX,
            max_depth: depth,
            max_nodes: u64::MAX,
        };
        let r = s.search(&mut pos, &limits);
        (r.best.to_uci(), r.score)
    }

    #[test]
    fn mate_in_one() {
        let (mv, score) = best("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 3);
        assert_eq!(mv, "a1a8");
        assert!(score > MATE_IN_MAX, "score {score}");
    }

    #[test]
    fn back_rank_mate_in_two() {
        // 1.Re8+ Rxe8 2.Rxe8#  (white rooks d1/e1, black king g8, pawns f7g7h7)
        let (_mv, score) = best("4r1k1/5ppp/8/8/8/8/5PPP/3RR1K1 w - - 0 1", 6);
        assert!(score > MATE_IN_MAX, "expected mate score, got {score}");
    }

    #[test]
    fn avoids_stalemate_when_winning() {
        // white must not stalemate; up a queen, should keep score high and legal
        let (mv, score) = best("7k/8/6K1/8/8/8/8/6Q1 w - - 0 1", 6);
        assert!(!mv.is_empty());
        assert!(score > 500, "score {score}");
    }

    #[test]
    fn recaptures_material() {
        // black just took a knight on c3 with the b-pawn; white recaptures
        let (mv, _) = best("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/2p2N2/PPPP1PPP/R1BQK2R w KQkq - 0 5", 8);
        assert_eq!(mv, "d2c3");
    }

    #[test]
    fn no_crash_on_bare_kings() {
        let (_mv, score) = best("8/8/4k3/8/8/4K3/8/8 w - - 0 1", 6);
        assert_eq!(score, 0);
    }
}
