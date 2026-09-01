"""Iterative-deepening alpha-beta over a material + piece-square evaluation.

Shipped in the zip. The value net on one CPU core is too slow, and too noisy,
to steer the deep search - letting it order moves or weigh into the evaluation
made the search measurably weaker. So the search runs entirely on the fast
hand-crafted evaluation (~15k nodes/s, depth 5-7 in the budget), and the
network decides only among the root moves the search rates as materially equal
(within tiebreak_cp of the best). That is most real positions, so the model
still materially drives the move played - it just cannot throw away material to
do it.

    move = AlphaBetaSearch(evaluator).run(board, deadline_monotonic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess

from evaluation import material_pst_cp
from inference import Evaluator

_MATE = 1_000_000.0
_INF = 2_000_000.0


@dataclass(frozen=True)
class AlphaBetaConfig:
    tiebreak_cp: float = 40.0  # net decides among root moves within this of the best
    value_tiebreak_cp: float = 60.0  # weight of the value head inside the tiebreak
    qsearch_captures: int = 8
    max_depth: int = 40
    use_net: bool = True


class _Timeout(Exception):
    pass


_EXACT, _LOWER, _UPPER = 0, 1, 2
_TTEntry = tuple[int, float, int, chess.Move | None]  # depth, score, flag, best move


class AlphaBetaSearch:
    def __init__(self, evaluator: Evaluator | None, config: AlphaBetaConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or AlphaBetaConfig()
        self._deadline = 0.0
        self._nodes = 0
        self._priors: dict[chess.Move, float] = {}
        self._root_scores: dict[chess.Move, float] = {}
        self._tt: dict[object, _TTEntry] = {}
        self._killers: dict[int, list[chess.Move]] = {}

    def run(self, board: chess.Board, deadline: float) -> chess.Move:
        self._deadline = deadline
        self._nodes = 0
        self._tt = {}
        self._killers = {}
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        use_net = self.config.use_net and self.evaluator is not None
        self._priors = {}
        child_value: dict[chess.Move, float] = {}
        if use_net:
            assert self.evaluator is not None
            self._priors, _ = self.evaluator.evaluate([board])[0]
            child_value = self._root_child_values(board, legal)

        self._root_scores = {}
        best = legal[0]
        for depth in range(1, self.config.max_depth + 1):
            try:
                # a timeout unwinds without popping; search a fresh copy each pass
                best = self._root(board.copy(), depth, best)
            except _Timeout:
                break

        if not use_net or not self._root_scores:
            return best
        return self._net_pick(child_value)

    def _net_pick(self, child_value: dict[chess.Move, float]) -> chess.Move:
        """Among root moves the search rates as materially equal, let the net choose."""
        cutoff = max(self._root_scores.values()) - self.config.tiebreak_cp
        contenders = [m for m, score in self._root_scores.items() if score >= cutoff]
        weight = self.config.value_tiebreak_cp
        return max(
            contenders,
            key=lambda m: 100.0 * self._priors.get(m, 0.0) - weight * child_value.get(m, 0.0),
        )

    def _root_child_values(
        self, board: chess.Board, legal: list[chess.Move]
    ) -> dict[chess.Move, float]:
        assert self.evaluator is not None
        children: list[chess.Board] = []
        for move in legal:
            board.push(move)
            children.append(board.copy(stack=False))
            board.pop()
        return {
            move: value
            for move, (_, value) in zip(legal, self.evaluator.evaluate(children), strict=True)
        }

    def _root(self, board: chess.Board, depth: int, prev_best: chess.Move) -> chess.Move:
        # full window at the root ply so every _root_scores entry is exact and can
        # be combined with the net value; deeper plies still prune normally
        best, best_score = prev_best, -_INF
        for move in self._ordered(board, prev_best):
            board.push(move)
            score = -self._negamax(board, depth - 1, -_INF, _INF)
            board.pop()
            self._root_scores[move] = score
            if score > best_score:
                best_score, best = score, move
        return best

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        self._nodes += 1
        if self._nodes % 256 == 0 and time.monotonic() >= self._deadline:
            raise _Timeout
        if board.is_checkmate():
            return -_MATE + float(board.ply())
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        key: object = board._transposition_key()
        alpha_original = alpha
        entry = self._tt.get(key)
        tt_move = entry[3] if entry else None
        if entry is not None and entry[0] >= depth:
            e_score, e_flag = entry[1], entry[2]
            if e_flag == _EXACT:
                return e_score
            if e_flag == _LOWER:
                alpha = max(alpha, e_score)
            else:
                beta = min(beta, e_score)
            if alpha >= beta:
                return e_score

        best, best_move = -_INF, None
        for move in self._ordered(board, tt_move, depth):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move):
                    self._remember_killer(depth, move)
                break

        flag = _EXACT if alpha_original < best < beta else (_LOWER if best >= beta else _UPPER)
        self._tt[key] = (depth, best, flag, best_move)
        return best

    def _remember_killer(self, depth: int, move: chess.Move) -> None:
        killers = self._killers.setdefault(depth, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

    def _quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        self._nodes += 1
        if self._nodes % 256 == 0 and time.monotonic() >= self._deadline:
            raise _Timeout
        stand_pat = float(material_pst_cp(board))
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        captures = sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: _mvv_lva(board, m),
            reverse=True,
        )[: self.config.qsearch_captures]
        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _ordered(
        self, board: chess.Board, first: chess.Move | None, depth: int = 0
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        killers = self._killers.get(depth, ())

        def key(move: chess.Move) -> tuple[bool, int, bool]:
            capture = _mvv_lva(board, move) if board.is_capture(move) else 0
            return (move == first, capture, move in killers)

        moves.sort(key=key, reverse=True)
        return moves


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    return victim * 10 - attacker
