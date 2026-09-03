# Optimisation backlog

Where strength can still come from, roughly ranked by Elo-per-hour. Numbers are
estimates for an engine currently around 1800-1900 on this hardware (1 core,
~20-40k nodes/s in Python). Everything here should be **SPRT-tested** before it
lands — we have repeatedly shipped "obvious" improvements that measured flat.

## 0. Measurement first (blocks everything else)

Without this we are guessing. Half a day, near-zero risk, unblocks the rest.

| item | notes |
|---|---|
| **cutechess-cli gauntlet** | Build a Linux/WSL `cutechess-cli` setup. Opponents: 3-4 fixed engines with published CCRL ratings near our level (e.g. a weak Stockfish `nodes=` ladder rung, Fruit, a small Maia). Gives a real anchored number. |
| **SPRT self-play** | `cutechess-cli` with `-sprt elo0=0 elo1=8 alpha=0.05 beta=0.05`, a balanced opening book (`8moves_v3.pgn` or similar), 8-16 concurrency at fast TC (10+0.1). Every change gets an SPRT vs the previous best. |
| **Tactical regression** | WAC (300), ECM, or Arasan suite. Track solve-count at fixed nodes each build. Catches pruning that's too aggressive. |
| **Local training box** | Set up `torch-directml` on the RX6600 so net training stops depending on free-tier quotas that wipe. |

## 1. Evaluation terms (biggest hand-eval lever)

The scalar tune (commit `1accad0`) showed the weights are near-optimal for the
current *structure*. Gains now come from **new terms**. All are midgame-weighted
via the existing phase interpolation. Watch eval cost — we lose ~1 ply per ~40%
eval slowdown.

| term | est. Elo | effort | risk | test |
|---|---|---|---|---|
| **King safety** (attacker count + weight table around king ring, open/half files near king, pawn-storm distance) | +40-80 | 1-2 d | high — easy to over/under-tune, causes wild sacs if wrong | SPRT + tactical suite; sanity-check on known attacking positions |
| **Mobility** (legal-ish move count per piece type, separate mg/eg weights, exclude squares attacked by enemy pawns) | +30-50 | 1 d | med — slows eval, needs attack tables | SPRT; profile nodes/s before+after |
| **Pawn structure** (isolated, doubled, backward, connected/phalanx, pawn islands) — cache by pawn-hash | +20-40 | 1 d | low | SPRT |
| **Threats** (piece attacked by lower-value piece, hanging pieces, restricted pieces) | +15-30 | half d | low | SPRT |
| **Piece placement** (knight outposts on holes, bad bishop blocked by own pawns, rook on 7th, rook behind passed pawn, trapped bishop/rook) | +15-30 | 1 d | low | SPRT |
| **Space** (safe squares behind own pawns in the centre files) | +5-15 | half d | low | SPRT |
| **Endgame scaling** (opposite-coloured-bishop draw factor, KRPKR drawishness, wrong-rook-pawn, KBNK drive already exists) | +10-30 | 1 d | med | EG test positions, no-throw check |
| **Pawn-hash + eval-hash tables** | +5-10 (speed) | half d | low — watch 2 GB cap | nodes/s |

## 2. Search

Current: TT (unbounded dict), 2 killers, butterfly history (no decay), MVV-LVA,
PVS, simple LMR, null-move R=3, RFP (d≤6), frontier futility (d≤3), check
extensions, aspiration. Quiescence d4 with delta pruning + a 1-ply recapture skip.

| item | est. Elo | effort | risk | test |
|---|---|---|---|---|
| **Search-param SPRT sweep** (RFP margin, futility margin, LMR base/divisor, null R formula, aspiration delta, qsearch depth) | +20-40 cumulative | ongoing, compute-bound | low | this is just SPRT time |
| **Proper SEE** (static exchange eval) for capture ordering + qsearch pruning + LMR of losing captures | +15-30 | 1 d | med — SEE bugs hurt | perft-style SEE unit tests, SPRT |
| **Bound the TT** (fixed-size, depth-preferred + aging replacement) | +5-15, and removes an OOM risk | half d | low | long-game memory check |
| **Countermove heuristic** + history decay/gravity + capture-history | +10-25 | half d | low | SPRT |
| **Late-move pruning** (skip quiets past a depth-scaled move count in non-PV) | +10-20 | half d | med | tactical suite + SPRT |
| **Singular extensions** (TT move fails to be beaten by a reduced search on the rest → extend) | +20-40 | 1-2 d | high | SPRT, tactical suite |
| **Internal iterative reduction** (no TT move at high depth → reduce instead of IID) | +5-15 | 2 h | low | SPRT |
| **Razoring** (drop to qsearch at low depth when static eval << alpha) | +5-10 | 2 h | med | tactical suite |
| **Quiescence upgrades** (generate check evasions, add checks at qdepth 0, qsearch TT probe) | +10-20 | half d | med | tactical suite |
| **History-based LMR** (reduce more when history is bad, less when good) | +10-20 | 2 h | low | SPRT |
| **Correction history** (adjust static eval by the running error between eval and search score for a pawn-structure/material bucket) | +10-25 | 1 d | med | SPRT |

## 3. Neural net

| item | est. Elo | effort | risk | notes |
|---|---|---|---|---|
| **Retrain to completion** (6-8 epochs, `small`, on local GPU) | baseline | 1 d compute | low | net #3 epoch-3 already beat net #1 (pol .456 / val .836); it was lost to a Colab wipe |
| **Value blend into eval** ("gated lazy eval": at PV nodes or depth ≥ N, `eval = w·netvalue_cp + (1-w)·handeval`, w tuned; hand-eval still dominates near material swings) | +50-150 | 2-3 d | high — needs calibrated value head + careful gating or it tanks nodes/s | the single biggest net lever; needs the value head calibration tightened first |
| **Policy for move ordering** (net priors order moves at root + PV nodes) | +20-50 | 1 d | med — costs net calls | gate to shallow plies only |
| **Bigger / better-trained net** (`medium` arch, more data, deeper SF labels, more epochs) | +20-60 | 2-3 d compute | low | value acc 0.84 → 0.86+ target |
| **Static int8 quant** (done, shelved — 1.9×, 4× smaller, max value err 0.45 needs better calibration) | 0 today | — | — | only pays off once the net is in the per-node path |
| **NNUE-style eval** (768→256×2→32→1, incremental accumulator, ~1-5 µs/eval, runs every node) | +150-300 | 4-6 d | high — big build | the *real* path past 2100 on this hardware; proper multi-day project |

## 4. Opening book & endgame

| item | est. Elo | effort | risk |
|---|---|---|---|
| **Deeper/wider book** (to move ~20-25, multiple replies per line, weight by observed win-rate not just SF-best) | +20-40 practical | 1 d | low |
| **Book from master games / strong self-play** rather than only Lichess-eval near-start positions | +10-30 | 1 d | low |
| **5-man syzygy** if the zip budget allows (WDL only ~1 GB — probably too big; confirm limit) | +5-15 | half d | low |
| **Book exit sanity** (never leave book into a known-bad structure; verify with a quick search) | +5-15 | 2 h | low |

## 5. Time management

Current: `min(time/max(20, 55-fullmove) + 300, time-500)`, panic < 4 s.

| item | est. Elo | effort | risk |
|---|---|---|---|
| **Spend by position** (more time when eval is unstable / PV changed last iteration / many near-equal root moves; less when one move dominates or in book) | +10-25 | half d | low |
| **Early exit on stable PV** (best move unchanged and margin large for 2+ iterations → move now, bank the time) | +5-15 | 2 h | low |
| **Increment-aware budgeting** (bank the 0.5 s/move, don't over-spend early) | +5-10 | 2 h | low |
| **Tune the budget curve** by SPRT | +5-15 | compute | low |

## 6. Longer shots (post-deadline)

- Compiled search core (Cython or C extension) — 3-8× nodes/s, +1-2 ply, but ABI/platform risk = forfeit if the `.so` doesn't load; needs their Python version pinned. Confirm the rules allow shipped binaries first.
- Multi-core is out (1 CPU core).
- Lazy SMP / pondering — not applicable.

## Suggested order for the week

1. **Day 1:** measurement infra (§0) — cutechess gauntlet + SPRT + tactical suite + local GPU.
2. **Day 2:** retrain net (background) ‖ eval: king safety + mobility.
3. **Day 3:** eval: pawn structure + threats + piece placement; SPRT everything from day 2.
4. **Day 4:** search-param SPRT sweep ‖ SEE + bound the TT + countermove/history.
5. **Day 5:** value-blend-into-eval with the retrained net (the big swing).
6. **Day 6:** book depth/width; time management.
7. **Day 7:** LMP + singular extensions + IIR; final SPRT gauntlet; pick the build.
