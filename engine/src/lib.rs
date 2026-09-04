//! Python bindings for the Chessathon engine.
//!
//! `best_move(fen, time_left_ms, ...)` runs an iterative-deepening search and
//! returns the chosen move in UCI. Everything heavy (movegen, eval, search)
//! lives in Rust; `agent.py` is a thin wrapper with a pure-Python fallback.

mod eval;
mod movegen;
mod perft;
mod position;
mod search;
mod see;
mod tables;
mod tt;
mod types;

use position::Position;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use search::{Limits, Searcher};
use std::sync::Mutex;
use types::Move;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// one searcher (and its TT) kept alive across calls within a process
struct Engine {
    searcher: Searcher,
}

fn engine() -> &'static Mutex<Engine> {
    use std::sync::OnceLock;
    static E: OnceLock<Mutex<Engine>> = OnceLock::new();
    E.get_or_init(|| {
        // touch the tables so the ~ms of magic generation happens at import
        tables::tables();
        Mutex::new(Engine {
            searcher: Searcher::new(128),
        })
    })
}

#[pyfunction]
fn version() -> String {
    format!("chessathon-engine {VERSION}")
}

/// Warm the attack tables and TT. Call once at import.
#[pyfunction]
fn init() -> bool {
    let guard = engine().lock().unwrap();
    drop(guard);
    true
}

fn parse(fen: &str) -> PyResult<Position> {
    Position::from_fen(fen).map_err(|e| PyValueError::new_err(format!("bad fen: {e}")))
}

/// Apply a list of UCI moves to a position, so the search sees the real game
/// history (for repetition detection).
fn apply_moves(pos: &mut Position, moves: &[String]) -> PyResult<()> {
    for uci in moves {
        let legal = pos.legal_moves();
        let mv = legal.iter().find(|m| m.to_uci() == *uci).copied();
        match mv {
            Some(m) => pos.make_move(m),
            None => return Err(PyValueError::new_err(format!("illegal history move {uci}"))),
        }
    }
    Ok(())
}

/// Pick a move. `budget_ms` is the soft target, `hard_ms` the absolute cap.
/// `moves` are UCI moves already played from `fen` to reach the position to search.
#[pyfunction]
#[pyo3(signature = (fen, budget_ms, hard_ms, moves=Vec::new(), max_depth=64))]
fn best_move(
    fen: &str,
    budget_ms: u64,
    hard_ms: u64,
    moves: Vec<String>,
    max_depth: i32,
) -> PyResult<String> {
    let mut pos = parse(fen)?;
    apply_moves(&mut pos, &moves)?;
    let mut guard = engine().lock().unwrap();
    let limits = Limits {
        soft_ms: budget_ms,
        hard_ms: hard_ms.max(budget_ms),
        max_depth: max_depth.clamp(1, 120),
        max_nodes: u64::MAX,
    };
    let r = guard.searcher.search(&mut pos, &limits);
    if r.best.is_none() {
        return Err(PyValueError::new_err("no legal move"));
    }
    Ok(r.best.to_uci())
}

/// Search to a fixed depth (for tests / analysis). Returns (uci, score_cp, depth, nodes).
#[pyfunction]
fn search_depth(fen: &str, depth: i32) -> PyResult<(String, i32, i32, u64)> {
    let mut pos = parse(fen)?;
    let mut guard = engine().lock().unwrap();
    let limits = Limits {
        soft_ms: u64::MAX,
        hard_ms: u64::MAX,
        max_depth: depth.clamp(1, 120),
        max_nodes: u64::MAX,
    };
    let r = guard.searcher.search(&mut pos, &limits);
    Ok((r.best.to_uci(), r.score, r.depth, r.nodes))
}

#[pyfunction]
fn evaluate_fen(fen: &str) -> PyResult<i32> {
    let pos = parse(fen)?;
    Ok(eval::evaluate(&pos))
}

#[pyfunction]
#[pyo3(name = "perft")]
fn perft_fen(fen: &str, depth: u32) -> PyResult<u64> {
    let mut pos = parse(fen)?;
    Ok(perft::perft(&mut pos, depth))
}

#[pyfunction]
fn legal_moves(fen: &str) -> PyResult<Vec<String>> {
    let pos = parse(fen)?;
    Ok(pos.legal_moves().iter().map(|m| m.to_uci()).collect())
}

/// Clear the transposition table (call between games).
#[pyfunction]
fn reset() {
    engine().lock().unwrap().searcher.tt.clear();
}

#[pymodule]
fn chessathon_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(search_depth, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_fen, m)?)?;
    m.add_function(wrap_pyfunction!(perft_fen, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(reset, m)?)?;
    let _ = Move::none();
    Ok(())
}
