//! Perft: count leaf nodes to a fixed depth. The move-generator's correctness net.

use crate::position::Position;

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for mv in moves {
        pos.make_move(mv);
        nodes += perft(pos, depth - 1);
        pos.unmake_move();
    }
    nodes
}

pub fn perft_divide(pos: &mut Position, depth: u32) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for mv in pos.legal_moves() {
        pos.make_move(mv);
        let n = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake_move();
        out.push((mv.to_uci(), n));
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(fen: &str, expected: &[u64]) {
        let mut pos = Position::from_fen(fen).unwrap();
        for (i, &e) in expected.iter().enumerate() {
            let d = i as u32 + 1;
            let got = perft(&mut pos, d);
            assert_eq!(got, e, "fen `{fen}` depth {d}: got {got}, want {e}");
        }
    }

    #[test]
    fn startpos() {
        check(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[20, 400, 8902, 197_281, 4_865_609],
        );
    }

    #[test]
    fn kiwipete() {
        check(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2039, 97_862, 4_085_603],
        );
    }

    #[test]
    fn position3() {
        check(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[14, 191, 2812, 43_238, 674_624],
        );
    }

    #[test]
    fn position4() {
        check(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9467, 422_333],
        );
    }

    #[test]
    fn position5() {
        check(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1486, 62_379, 2_103_487],
        );
    }

    #[test]
    fn position6() {
        check(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2079, 89_890, 3_894_594],
        );
    }

    #[test]
    fn startpos_deep() {
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(perft(&mut pos, 6), 119_060_324);
    }

    #[test]
    fn kiwipete_deep() {
        let mut pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&mut pos, 5), 193_690_690);
    }

    #[test]
    fn ep_pin_edge_cases() {
        // en passant that would expose the king along the rank - must be rejected.
        // Reference counts cross-checked against python-chess.
        check("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1", &[6, 78, 528, 8_288, 55_203]);
        check("8/8/8/8/k2Pp2R/8/8/7K b - d3 0 1", &[6, 78, 528, 8_239, 55_125]);
    }
}
