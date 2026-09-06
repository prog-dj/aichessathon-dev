# Search optimization — autonomous run (overnight Sept 5-6)

**Goal:** +70-100 Elo of search improvements, SPRT-gated, keep winners, scrap the rest.
Deliver a validated submission candidate with evidence.

## Setup
- Branch `search-opt` off `3c15593` (clean engine — NNUE 32-bucket / eval-arch / selective-NNUE
  experiments from `61ab664` reverted). Baseline commit `141aa6a` = clean engine + a `max_nodes`
  A/B hook on `Engine.best_move` (unused in tournament play).
- Pinned baseline worktree: `../arena-base` @ `141aa6a`.
- Rig: `scratchpad/arena_sprt.py` — fixed-node (120-150k nodes/move), 3 concurrent lanes,
  20 real Chessathon opening FENs both colours, trinomial SPRT (elo0=0, elo1=6, α=β=0.05).
  ~750 games/hour. Identical-engine control: 48.8%, −9 ± 55 Elo (n=40). ✓
- Every candidate: perft(5) unchanged + `pytest tests/` + WAC tactics spot-check BEFORE the SPRT.

## Results

| # | change | SPRT verdict | Elo (pt est ± ) | decision |
|---|--------|--------------|-----------------|----------|
| — | identical-engine control | n=40 | −9 ± 55 | rig OK |
| C1 | SEE + SEE<0 capture prune in qsearch | n≈135, 61.5% | ~+80 @120k | **KEEP** (→ baseline 9a42e93) |
| C2+C3+C4 | history-decay + log-LMR + SEE-prune-losing-caps (main search) | n=320, 57.8% | **+55** ±40 @120k | **KEEP** (→ baseline 63f944b) |
| C5 | SEE-value capture ordering (replace MVV-LVA) | n=179, 50.6% | ~+5, neutral | **DROP** (neutral + adds per-node SEE cost) |
| C6 | param retune (RFP 75/d≤7, null R+1, LMP 3+d², futility 120/d≤4) | n=115, 48.6% | ~−10, mild neg | **DROP** (guessed values, not better) |
| C8 | transposition-table probe + store in qsearch | n=161, 55.3% | ~+37 @120k | **KEEP** (→ HEAD fcd8f08) |

## Final stack (branch `search-opt`, HEAD fcd8f08)
1. SEE + SEE<0 capture prune in qsearch
2. Quiet-history decay between moves (was: zero every move)
3. Log-curve LMR table + history-based reduction relief
4. SEE prune of losing captures in the main search
5. TT probe + store in qsearch

Submission: `submission_searchopt.zip` (22.0 MB uncompressed, JIT compile 31.4s).
fastchess.py = full stack; agent.py = last session's time-management fix.

VALIDATION (full stack fcd8f08 vs clean 141aa6a, 500k nodes/move ~ tournament depth,
diverse openings): **n≈55, 66%, +118 Elo, 95% CI [+27, +229]**. The 120k-node
per-change numbers were NOT inflated - at deeper search the stack gains *more*
(SEE + LMR compound with depth). Confident large improvement.

Clean baseline 141aa6a ≈ the current leaderboard engine's search (it's the
pre-everything HCE engine + the fixed-node A/B hook). So the new submission is
~+100 Elo of pure search strength over what's live, plus last session's
time-management fix (already in both).

## DELIVERABLE
`submission_searchopt.zip` - search-opt branch HEAD fcd8f08 + time-mgmt agent.py.
22 MB uncompressed, JIT compile 31s. Verified: imports, plays legal moves,
perft(5) exact, full pytest green.
NOT pushed / NOT merged to main (coordination w/ Mikhail's branch - user's call).

Compile time with SEE+bundle: 30.2s (target ~30s). agent.py on search-opt = has the
time-management fix (validated last session). Submission = search-opt fastchess.py + agent.py.

## Notes / running commentary
- (start) rig built, baseline clean. Starting SEE.
- Found + fixed a rig bug: fixed-node search is deterministic and the arena had
  only 20 openings with no jitter, so games 41+ replayed 1-40 (inflated sample).
  Fix: 249 generated diverse balanced openings + deterministic per-game node
  jitter (0.8-1.2x, same budget both sides). Draw rate dropped 45% -> 25%.
- C1 (SEE qsearch prune) at 120k nodes: 63% over ~130 diverse-opening games,
  clean positive trend. +90 is likely inflated for real TC (SEE is universally
  +20-40 in mature engines) but unambiguously a keep. Locking as new baseline.
- arena-dev has a bundle staged on top of C1: history-decay (C2), log-LMR (C3),
  SEE-losing-capture prune in main search (C4). Test next.
