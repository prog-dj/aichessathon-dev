# chessathon-engine

Bitboard chess move generation, evaluation and search for the AI Chessathon
agent. Built as a Python extension with [maturin](https://www.maturin.rs/) /
[pyo3](https://pyo3.rs/); `agent.py` imports it and falls back to a pure-Python
engine if the wheel is unavailable.

```python
import chessathon_engine as ce
ce.init()
ce.best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2000, 2500)
```

Local dev:

```
cd engine
maturin develop --release          # build + install into the active venv
cargo test --release                # perft suite
```
