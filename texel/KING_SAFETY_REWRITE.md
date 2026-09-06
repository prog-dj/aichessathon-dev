# King-safety — ATTEMPTED AND DIAGNOSED (read this before trying again)

## RESULT: the rewrite was built, measured, and abandoned. Root cause found.

The accumulate-linearly / threshold / single-nonlinearity design in this spec was
implemented in full (attacker weights, ring attacks, safe checks via per-piece
enemy attack maps, distance-based shelter, no-queen discount, ring defenders).
It was correct: perft(5) exact, **800/800 colour-symmetry**, and the eval-change
distribution was healthy (median 11cp, p90 83cp, p99 239cp — versus the failed
gated attempt's mean 188cp with 781cp cliffs).

It still degraded eval quality. The decisive experiment was an **ablation**,
measuring correlation with Stockfish on the subset of positions where the term
actually fires:

| config | delta corr vs fcd8f08 |
|---|---|
| att+ring only (**identical inputs to the old term, new curve**) | **-0.051** |
| + safe checks only | -0.048 |
| + shelter only | -0.049 |
| both (full) | -0.025 |

**The control degrades as much as everything else.** The new signals are not the
problem — they recover half the loss. **Reshaping the curve is what breaks it.**

A 12-point sweep over scale (0.35/0.6/1.0) x threshold (100/250) x divisor
(400/900) found *no* configuration with a positive delta. Best was -0.025.

## Why: the eval is calibrated *around* the old curve
`_king_danger(u) = min(u,40)**2 * 11 // 16` maps units 0..40 onto 0..1100cp.
Every other eval term (PST, material, mobility, pawn structure) has been tuned
and evolved against that specific king-danger scale. Change the curve's shape and
the whole eval's internal balance shifts, even with identical inputs. Add signal
into the old curve and the quadratic amplifies it by hundreds of cp.

**King safety cannot be fixed in isolation on this engine.** It is entangled with
the rest of the eval's calibration.

## What would actually be required
Re-tune the ENTIRE eval (material, PST, mobility, pawns, king safety) *jointly*
with the new king-safety structure. That needs:
1. A better tuning target than Lichess SF-cp. SF's eval is holistic and depth-
   informed; tuning against it shrinks king-safety weights to nothing (measured)
   and the harness only moves val-MSE 0.0526 -> 0.0509. Use **game results from
   self-play** instead (generate ~100k+ games, label positions by outcome).
2. PST unfrozen in the tune (768 params) so the eval can rebalance.
3. Many SPRT rounds.
That is a multi-day project, not a term-add.

## Score so far: 5 attempts, 0 successes
pre-session revert; unconditional shelter (-98); Texel-tuned shelter (weights
collapsed); attacker-gated shelter (W1 D2 L15, ~-250); full rewrite (ablation
above, not worth an SPRT). The engine's in-band (+/-300cp) correlation with SF is
**0.177** and none of these moved it.

---
# Original spec (kept for reference)

## Goal
Replace the king-danger term in `fastchess.py:evaluate_hce` with a Stockfish-style
accumulate-then-one-nonlinearity design. Target +40–80 Elo. It's the recurring
loss pattern (rounds 31, 32: engine opens its own king / pushes shield pawns and
gets mated; eval doesn't see it).

## Current term (the thing to replace)
In `evaluate_hce`, per piece in the loop:
```
rh = popcount(att & ring)                 # ring = KING_ATT[enemy_ksq] | enemy_ksq
if rh > 0:
    w = 2/2/3/5 for N/B/R/Q
    danger[them] += w * rh
```
then:
```
_king_danger(u) = min(u,40)**2 * 11 // 16     # QUADRATIC
mg += _king_danger(danger[1]) - _king_danger(danger[0])
```
Operating range: danger 0–40 → score 0–1100 cp. Way too hot.

## Why every add-on has failed (4 attempts)
Adding ANY term to `danger[]` before the square amplifies violently:
danger 15→25 makes the score jump 155→430 cp (+275) on ~17% of positions →
eval destabilises → move choice collapses.
- unconditional shelter: −98 Elo
- Texel-tuned shelter: weights shrank to ~0 (SF-cp is a bad target for king safety)
- attacker-gated shelter (fires only at ≥2 attackers): W1 D2 L15, ~−250 Elo

## Target design
Accumulate `kd` **linearly**, apply the nonlinearity **once**, with a divisor
tuned so the realistic range maps to sane cp.

```
# per enemy attacker of the king ring (in the piece loop):
kd += ATT_W[pt] * n_ring_squares_attacked          # ATT_W ~ [ ~14 N, ~10 B, ~8 R, ~40 Q ] (SF-ish, scale to taste)
attacker_count += 1

# once, per king, after the loop (only if attacker_count >= 2, else kd_term = 0):
kd += ring_squares_attacked_total * 8
kd += safe_checks_N * 40 + safe_checks_B * 25 + safe_checks_R * 30 + safe_checks_Q * 25
kd += shelter_holes * SH + kfile_open * OF + flank_open * FF     # small, e.g. 6/25/15
kd -= friendly_ring_defenders * 6
kd -= known_pawn_shield_bonus                                    # pawn directly in front

king_danger_cp = max(0, kd) * max(0, kd) / DIV       # DIV ~ 700–1000; tune so kd~25 -> ~80cp, kd~50 -> ~350cp
mg += bd_cp - wd_cp
```

**safe_checks**: squares adjacent-ish to the king from which an enemy piece could
give check, that our non-king pieces don't defend. Needs per-piece-type enemy
attack bitboards accumulated in the loop (`att_by_type[them][pt] |= att`).
`knight_check_sq = KNIGHT_ATT[ksq] & att_by_type[them][N] & ~our_attacks & ~their_own`
`bishop_check_sq = bishop_attacks(ksq, occ) & (att_by_type[them][B] | att_by_type[them][Q]) & ~our_attacks`
etc. This is the term that actually models "gets mated" and is where the Elo is.

## Calibration protocol (SPRT, NOT Texel)
- Texel against SF-cp shrinks king safety to nothing — SF's king eval is holistic.
  Use `texel/` only as a sanity check (does eval stay ~consistent on quiet positions).
- Start CONSERVATIVE (DIV high ~1000, small shelter weights). SPRT vs `fcd8f08`.
- If neutral → scale weights up 1.3x, re-SPRT. If negative → DIV up / weights down.
- Iterate to the level that's clearly positive. 3–5 SPRT rounds expected.

## Test rig (already set up)
```
# baseline worktree (current best engine): C:/Users/44783/AI_Chessathon/arena-base  @ fcd8f08
# candidate worktree:                       C:/Users/44783/AI_Chessathon/arena-dev  (on branch `dev`, keep == search-opt + your change)
# clean pre-everything (headline compares): C:/Users/44783/AI_Chessathon/arena-clean @ 141aa6a

python scratchpad/arena_sprt.py <CAND_DIR> <BASE_DIR> 150000 360 3 <tag>
# fixed 150k nodes/move, 3 lanes, 249 diverse openings, ~15 min for 360 games.
# read the score CI directly (elo1=6 makes the formal SPRT bound unreachable in that many games).
# identical-engine control was 48.8% / -9 Elo.
```
`arena_sprt.py` DRIVER also supports negative nodes = ms/move (time mode) for a
final validation.

## Sanity gates before any SPRT
- `perft_fen(startpos, 5) == 4865609`
- `pytest tests/ -q` all green
- eval on ~10 quiet fianchetto / luft positions unchanged or nearly so vs fcd8f08
- eval on a few "king under real attack" positions is more negative for the
  attacked side (that's the point) but not by >~400cp

## Files
- `fastchess.py:evaluate_hce` (the term) + helper njit functions near `_king_danger`
- Mirror into `texel/features.py` + `texel/model.py` ONLY if you want the harness
  sanity check; not required.
- Commit on `search-opt`; if SPRT fails, `git revert HEAD` (engine goes back to
  fcd8f08 byte-identical — verified pattern).

## Shipped state (don't disturb)
`submission_searchopt.zip` = `fcd8f08` = SEE + log-LMR + qsearch-TT + history-decay,
validated +85 Elo vs 141aa6a. King safety rides on top of this or not at all.
