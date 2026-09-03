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

from evaluation import PIECE_VALUE, evaluate_cp, material_pst_white, mpst_quiet_delta
from inference import Evaluator

_MATE = 1_000_000.0
_INF = 2_000_000.0


@dataclass(frozen=True)
class AlphaBetaConfig:
    tiebreak_cp: float = 40.0  # net decides among root moves within this of the best
    value_tiebreak_cp: float = 60.0  # weight of the value head inside the tiebreak
    qsearch_depth: int = 4  # cap on quiescence plies
    delta_margin_cp: float = 120.0  # skip a capture that cannot get near alpha
    contempt_cp: float = 30.0  # a draw counts this much against us - play won games on
    rfp_margin_cp: float = 75.0  # reverse-futility margin per ply
    futility_margin_cp: float = 100.0  # frontier futility margin per ply
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
        self._depth_reached = 0
        self._root_ply = 0
        self._priors: dict[chess.Move, float] = {}
        self._root_scores: dict[chess.Move, float] = {}
        self._root_exact: set[chess.Move] = set()
        self._tt: dict[object, _TTEntry] = {}
        self._killers: dict[int, list[chess.Move]] = {}
        self._history: dict[tuple[bool, int, int], int] = {}

    def run(self, board: chess.Board, deadline: float) -> chess.Move:
        self._deadline = deadline
        self._nodes = 0
        self._tt = {}
        self._killers = {}
        self._history = {}
        self._root_ply = board.ply()
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("search called on a position with no legal moves")
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
        self._root_exact = set()
        best = legal[0]
        self._depth_reached = 0
        score = 0.0
        for depth in range(1, self.config.max_depth + 1):
            try:
                # a timeout unwinds without popping; search a fresh copy each pass
                best, score = self._aspirate(board, depth, best, score)
                self._depth_reached = depth
            except _Timeout:
                break

        if not use_net or not self._root_scores:
            return best
        return self._net_pick(best, child_value)

    def _aspirate(
        self, board: chess.Board, depth: int, best: chess.Move, prev_score: float
    ) -> tuple[chess.Move, float]:
        if depth <= 3:
            return self._root(board.copy(), depth, best, -_INF, _INF)
        delta = 45.0
        while True:
            low, high = prev_score - delta, prev_score + delta
            best, score = self._root(board.copy(), depth, best, low, high)
            if low < score < high:
                return best, score
            delta *= 3.5
            if delta > 1500.0:  # window keeps failing: give up and search wide
                return self._root(board.copy(), depth, best, -_INF, _INF)

    def _net_pick(self, best: chess.Move, child_value: dict[chess.Move, float]) -> chess.Move:
        """Among root moves with an exact score near the best, let the net choose."""
        cutoff = self._root_scores[best] - self.config.tiebreak_cp
        contenders = [
            m for m in self._root_exact if self._root_scores.get(m, -_INF) >= cutoff
        ]
        if not contenders:
            return best
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

    def _root(
        self, board: chess.Board, depth: int, prev_best: chess.Move, alpha: float, beta: float
    ) -> tuple[chess.Move, float]:
        ordered = self._ordered(board, prev_best)
        mw = material_pst_white(board)
        best, best_score = prev_best, -_INF
        window = alpha
        for index, move in enumerate(ordered):
            delta = mpst_quiet_delta(board, move)
            child_mw = None if delta is None else mw + delta
            board.push(move)
            if index == 0:
                score = -self._negamax(board, depth - 1, -beta, -window, child_mw)
            else:
                score = -self._negamax(board, depth - 1, -window - 1, -window, child_mw)
                if window < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -window, child_mw)
            board.pop()
            self._root_scores[move] = score
            if score > best_score:
                best_score, best = score, move
            window = max(window, score)

        # second pass, only for the net tiebreak: exact scores for moves near the best
        self._root_exact = {best}
        if self.evaluator is not None and alpha <= best_score <= beta:
            cutoff = best_score - self.config.tiebreak_cp
            for move in ordered:
                if move != best and self._root_scores.get(move, -_INF) >= cutoff:
                    board.push(move)
                    self._root_scores[move] = -self._negamax(board, depth - 1, -_INF, _INF, None)
                    board.pop()
                    self._root_exact.add(move)
        return best, best_score

    def _negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, mw: float | None
    ) -> float:
        self._nodes += 1
        if self._nodes % 256 == 0 and time.monotonic() >= self._deadline:
            raise _Timeout
        if board.is_insufficient_material() or (
            board.halfmove_clock >= 8 and (board.is_repetition(3) or board.is_fifty_moves())
        ):
            return self._draw_score(board)
        if mw is None:
            mw = material_pst_white(board)
        if depth <= 0:
            return self._quiescence(board, alpha, beta, 0, mw)

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

        moves = self._ordered(board, tt_move, depth)
        in_check = board.is_check()
        if not moves:  # no legal move: checkmate if in check, else stalemate
            return -_MATE + float(board.ply()) if in_check else self._draw_score(board)

        non_pv = beta - alpha == 1
        static_eval = None if in_check else float(evaluate_cp(board, material_white=mw))

        # reverse futility: the position is already so good a search is not needed
        if (
            static_eval is not None
            and non_pv
            and depth <= 6
            and abs(beta) < _MATE - 1000
            and static_eval - self.config.rfp_margin_cp * depth >= beta
        ):
            return static_eval

        # null-move pruning: if passing still beats beta, this node is winning enough to skip
        if depth >= 3 and not in_check and non_pv and _has_non_pawn_material(board):
            board.push(chess.Move.null())
            null_score = -self._negamax(board, depth - 3, -beta, -beta + 1, mw)
            board.pop()
            if null_score >= beta and abs(null_score) < _MATE - 1000:
                return beta

        futile = (
            static_eval is not None
            and depth <= 3
            and static_eval + self.config.futility_margin_cp * depth <= alpha
        )
        deep = board.ply() - self._root_ply < 2 * max(depth, self._depth_reached) + 4
        best, best_move = -_INF, None
        for index, move in enumerate(moves):
            capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            quiet = not capture and not gives_check
            if futile and index > 0 and quiet and best > -_MATE + 1000:
                continue  # frontier futility: this quiet move cannot lift alpha
            delta = None if not quiet else mpst_quiet_delta(board, move)
            child_mw = None if delta is None else mw + delta
            board.push(move)
            # check extension: follow forcing lines a ply further
            child_depth = depth if (gives_check and deep) else depth - 1
            if index == 0:
                score = -self._negamax(board, child_depth, -beta, -alpha, child_mw)
            else:
                # late-move reduction: search likely-bad quiet moves shallower first
                reduction = 0
                if index >= 4 and depth >= 3 and quiet and not in_check:
                    reduction = 1 + (index >= 10 and depth >= 6)
                score = -self._negamax(board, child_depth - reduction, -alpha - 1, -alpha, child_mw)
                if score > alpha:  # scout beat alpha: re-search at full depth and window
                    score = -self._negamax(board, child_depth, -beta, -alpha, child_mw)
            board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    self._remember_killer(depth, move)
                    self._history[(board.turn, move.from_square, move.to_square)] = (
                        self._history.get((board.turn, move.from_square, move.to_square), 0)
                        + depth * depth
                    )
                break

        flag = _EXACT if alpha_original < best < beta else (_LOWER if best >= beta else _UPPER)
        self._tt[key] = (depth, best, flag, best_move)
        return best

    def _draw_score(self, board: chess.Board) -> float:
        """A draw, seen from the side to move: negative when it is our turn (so we
        play a won game on), positive when it is the opponent's."""
        ours = (board.ply() - self._root_ply) % 2 == 0
        return -self.config.contempt_cp if ours else self.config.contempt_cp

    def _remember_killer(self, depth: int, move: chess.Move) -> None:
        killers = self._killers.setdefault(depth, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

    def _quiescence(
        self, board: chess.Board, alpha: float, beta: float, qdepth: int, mw: float | None
    ) -> float:
        self._nodes += 1
        if self._nodes % 256 == 0 and time.monotonic() >= self._deadline:
            raise _Timeout
        stand_pat = float(evaluate_cp(board, material_white=mw))
        if stand_pat >= beta or qdepth >= self.config.qsearch_depth:
            return stand_pat
        alpha = max(alpha, stand_pat)

        opponent = not board.turn
        captures = sorted(
            board.generate_legal_captures(), key=lambda m: _mvv_lva(board, m), reverse=True
        )
        for move in captures:
            victim = PIECE_VALUE[board.piece_type_at(move.to_square) or chess.PAWN]
            if stand_pat + victim + self.config.delta_margin_cp < alpha:
                continue  # delta pruning: even winning this piece won't reach alpha
            attacker = PIECE_VALUE.get(board.piece_type_at(move.from_square) or 0, 0)
            if 0 < victim < attacker and board.is_attacked_by(opponent, move.to_square):
                continue  # loses material on the recapture: skip
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, qdepth + 1, None)
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
        turn = board.turn
        history = self._history

        def key(move: chess.Move) -> tuple[bool, int, bool, int]:
            capture = _mvv_lva(board, move) if board.is_capture(move) else 0
            hist = history.get((turn, move.from_square, move.to_square), 0)
            return (move == first, capture, move in killers, hist)

        moves.sort(key=key, reverse=True)
        return moves


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    return victim * 10 - attacker


def _has_non_pawn_material(board: chess.Board) -> bool:
    ours = board.occupied_co[board.turn]
    return bool(ours & (board.knights | board.bishops | board.rooks | board.queens))
