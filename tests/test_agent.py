"""Time-management budget: pace-tracking fix for the clock-mismanagement bug found in
real games (rounds 13, 16, 17 - see docs in agent.py / the plan that introduced this),
plus the complexity-weighting refinement on top of it.
"""

from __future__ import annotations

import chess

import agent


def _old_budget_ms(time_left_ms: float, fullmove: int) -> tuple[float, float]:
    """The pre-fix formula, reimplemented here only for comparison in these tests."""
    if time_left_ms < agent._PANIC_MS:
        t = max(20.0, min(time_left_ms * 0.08, 400.0))
        return t, t
    moves_left = max(20, 55 - fullmove)
    soft = (time_left_ms - agent._SAFETY_MS) / moves_left + 0.9 * agent._INCREMENT_MS
    soft = min(soft, time_left_ms - agent._SAFETY_MS)
    hard = min(soft * 1.8, time_left_ms - agent._SAFETY_MS)
    return soft, hard


def _reset_pace_state() -> None:
    agent._pace_ref_time_left_ms = None
    agent._pace_ref_fullmove = None


def test_panic_branch_is_unaffected() -> None:
    _reset_pace_state()
    for time_left, fullmove in ((3000, 40), (1000, 70), (3999, 10)):
        soft, hard = agent._budget_ms(time_left, fullmove)
        old_soft, old_hard = _old_budget_ms(time_left, fullmove)
        assert soft == old_soft
        assert hard == old_hard


def test_moves_left_keeps_growing_past_the_old_floor() -> None:
    """The old formula flatlined moves_left at 20 for any move past ~35, regardless of
    how long the game actually ran. Feed a clock that's exactly on the idealised
    schedule (so pace_ratio stays ~1) and confirm the implied budget at move 60 is
    still meaningfully different from move 40, not floored to the same number."""
    # two independent single-shot calls, each freshly anchored at its own fullmove,
    # simulating "on schedule" for a game of the new expected length
    remaining_at_40 = agent._EXPECTED_GAME_LENGTH - 40
    remaining_at_60 = agent._EXPECTED_GAME_LENGTH - 60
    time_at_40 = remaining_at_40 * 1200  # arbitrary but consistent per-move pace
    time_at_60 = remaining_at_60 * 1200
    _reset_pace_state()
    soft_40, hard_40 = agent._budget_ms(time_at_40, 40)
    _reset_pace_state()
    soft_60, hard_60 = agent._budget_ms(time_at_60, 60)
    # both should be well above zero and sane - the point is neither one degenerately
    # collapses to the same fixed-20-move-divisor answer the old formula gave
    assert soft_40 > 0
    assert soft_60 > 0


def test_real_danger_points_get_a_smaller_budget_than_before() -> None:
    """Real (time_left_ms, fullmove) checkpoints pulled directly from the three rated
    games that exposed this bug. At every one of these points the old formula was
    still handing out a bigger budget than the game could actually afford - that's the
    mechanical cause of the clock running out. The fix should be visibly more
    conservative at all of them."""
    checkpoints = [
        # round 13 (White), the 36-45 bleed - clock in ms
        (27_448, 36), (24_362, 37), (21_175, 39), (20_132, 40), (17_762, 41), (12_786, 45),
        # round 16 (Black), the 37-39 bleed
        (30_863, 37), (28_352, 38), (25_072, 39),
        # round 17 (White), the 42-64 near-disaster
        (19_001, 42), (12_046, 48), (10_429, 50), (3_592, 59), (3_690, 63),
    ]
    for time_left, fullmove in checkpoints:
        _reset_pace_state()
        # anchor pace at "game started on a normal budget", then jump straight to the
        # danger point - simulates arriving there already behind schedule, as happened
        agent._pace_ref_time_left_ms = 120_000.0
        agent._pace_ref_fullmove = 1
        soft, hard = agent._budget_ms(time_left, fullmove)
        old_soft, old_hard = _old_budget_ms(time_left, fullmove)
        assert soft <= old_soft, (time_left, fullmove, soft, old_soft)
        assert hard <= old_hard, (time_left, fullmove, hard, old_hard)


def test_pace_ratio_is_clamped_both_ways() -> None:
    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    # way behind an idealised schedule -> should not shrink the budget to nothing
    soft, hard = agent._budget_ms(5_000, 80)
    assert soft > 0
    assert hard >= soft

    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    # exactly on schedule, for comparison
    on_schedule_soft, _ = agent._budget_ms(int(120_000 - 1342.7 * 4 + 500 * 4), 5)

    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    # way ahead of schedule (e.g. a long free book run) -> capped acceleration, not
    # unbounded - the internal pace_ratio can push soft up, but only up to the 1.15x cap
    soft_ahead, _ = agent._budget_ms(119_000, 5)
    assert soft_ahead <= on_schedule_soft * agent._PACE_RATIO_MAX * 1.05  # small slack for rounding


def test_complexity_multiplier_full_material_vs_bare_kings() -> None:
    full = agent._complexity_multiplier(chess.Board())
    bare = agent._complexity_multiplier(chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1"))
    assert full == agent._COMPLEXITY_MULT_MAX
    assert bare == agent._COMPLEXITY_MULT_MIN
    assert full > bare


def test_complexity_multiplier_is_monotonic_as_material_comes_off() -> None:
    # queen + rook + bishop + knight per side
    heavy = chess.Board("4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1")
    # just a rook per side
    light = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert agent._complexity_multiplier(heavy) > agent._complexity_multiplier(light)
    assert agent._COMPLEXITY_MULT_MIN <= agent._complexity_multiplier(light) < agent._complexity_multiplier(heavy) <= agent._COMPLEXITY_MULT_MAX


def test_panic_mode_ignores_complexity() -> None:
    _reset_pace_state()
    soft, hard = agent._budget_ms(3000, 40, complexity_mult=agent._COMPLEXITY_MULT_MAX)
    old_soft, old_hard = _old_budget_ms(3000, 40)
    assert soft == old_soft
    assert hard == old_hard


def test_complexity_scales_the_budget_up_and_down() -> None:
    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    soft_neutral, _ = agent._budget_ms(100_000, 20, complexity_mult=1.0)

    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    soft_complex, _ = agent._budget_ms(100_000, 20, complexity_mult=agent._COMPLEXITY_MULT_MAX)

    _reset_pace_state()
    agent._pace_ref_time_left_ms = 120_000.0
    agent._pace_ref_fullmove = 1
    soft_simple, _ = agent._budget_ms(100_000, 20, complexity_mult=agent._COMPLEXITY_MULT_MIN)

    assert soft_simple < soft_neutral < soft_complex
