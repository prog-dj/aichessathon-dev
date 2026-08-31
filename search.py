"""Batched PUCT search. Shipped in the zip.

A tree search guided by the network: the policy head supplies the prior over
moves, the value head scores leaves, and PUCT balances "this move looks good"
against "this move is under-explored". One network call evaluates a whole batch
of leaves, with virtual loss keeping the batched descents apart.

Tactical insurance on top of the plain algorithm: every capture and every
checking move is forced into the tree even if the policy ignores it, so a
forced refutation the policy misses still gets searched.

    search = PuctSearch(evaluator)
    move = search.run(board, deadline_monotonic)

``board`` should carry its full move history so repetitions and the fifty-move
rule are seen inside the tree.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess

from inference import Evaluator


@dataclass(frozen=True)
class PuctConfig:
    c_puct: float = 1.25
    batch_size: int = 8
    fpu_reduction: float = 0.5  # value penalty for an unvisited child
    virtual_loss: float = 1.0
    max_children: int = 12  # widen only to the top policy moves (+ forced tactics)
    tactical_floor: float = 0.02  # minimum prior handed to a forced capture/check
    contempt: float = 0.0  # draws count this much against the side to move at the root
    max_sims: int = 1_000_000


class Node:
    __slots__ = ("children", "prior", "value_sum", "visits")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)


class PuctSearch:
    def __init__(self, evaluator: Evaluator, config: PuctConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or PuctConfig()

    def run(self, board: chess.Board, deadline: float) -> chess.Move:
        root = Node(0.0)
        root_priors, _ = self.evaluator.evaluate([board])[0]
        self._expand(root, board, root_priors)
        if not root.children:
            raise ValueError("search called on a position with no legal moves")
        if len(root.children) == 1:
            return next(iter(root.children))

        sims = 0
        while sims < self.config.max_sims and time.monotonic() < deadline:
            pending: list[tuple[list[Node], chess.Board]] = []
            for _ in range(self.config.batch_size):
                path, leaf_board = self._descend(root, board)
                sims += 1
                terminal = self._terminal_value(leaf_board)
                if terminal is not None:
                    self._backup(path, self._with_contempt(terminal, len(path)))
                else:
                    pending.append((path, leaf_board))
            if pending:
                results = self.evaluator.evaluate([lb for _, lb in pending])
                for (path, leaf_board), (child_priors, value) in zip(pending, results, strict=True):
                    self._expand(path[-1], leaf_board, child_priors)
                    self._backup(path, value)

        return max(root.children.items(), key=lambda item: item[1].visits)[0]

    def _descend(self, root: Node, board: chess.Board) -> tuple[list[Node], chess.Board]:
        node = root
        node.visits += 1
        leaf_board = board.copy()
        path = [node]
        while node.expanded:
            move, child = self._best_child(node)
            leaf_board.push(move)
            child.visits += 1
            # virtual loss: pretend this child just lost, so the other batched
            # descents in this round pick something else
            child.value_sum += self.config.virtual_loss
            path.append(child)
            node = child
        return path, leaf_board

    def _best_child(self, parent: Node) -> tuple[chess.Move, Node]:
        sqrt_parent = math.sqrt(parent.visits)
        # unvisited children are scored from the parent's own value, slightly reduced
        first_play = parent.q - self.config.fpu_reduction
        best_move: chess.Move | None = None
        best_child: Node | None = None
        best_score = -math.inf
        for move, child in parent.children.items():
            q = -child.q if child.visits else first_play
            u = self.config.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        assert best_move is not None and best_child is not None
        return best_move, best_child

    def _expand(self, node: Node, board: chess.Board, priors: dict[chess.Move, float]) -> None:
        if node.children:
            return
        legal = list(board.legal_moves)
        forced = {m for m in legal if board.is_capture(m) or board.gives_check(m)}
        # widen only to the strongest policy moves, but never drop a forced tactic
        top = sorted(legal, key=lambda m: priors.get(m, 0.0), reverse=True)
        keep = set(top[: self.config.max_children]) | forced
        floor = self.config.tactical_floor
        weights = {
            move: max(priors.get(move, 0.0), floor if move in forced else 0.0) for move in keep
        }
        total = sum(weights.values()) or 1.0
        node.children = {move: Node(weight / total) for move, weight in weights.items()}

    def _backup(self, path: list[Node], leaf_value: float) -> None:
        # leaf_value is from the point of view of the side to move at the leaf
        value = leaf_value
        for node in reversed(path):
            node.value_sum += value
            value = -value
        for node in path[1:]:  # undo the virtual loss each descended node took
            node.value_sum -= self.config.virtual_loss

    @staticmethod
    def _terminal_value(board: chess.Board) -> float | None:
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_repetition(3)
            or board.can_claim_fifty_moves()
        ):
            return 0.0
        return None

    def _with_contempt(self, terminal: float, path_len: int) -> float:
        """Nudge draw scores against the root player so a won game gets played on."""
        if terminal != 0.0 or not self.config.contempt:
            return terminal
        leaf_is_root_side = (path_len - 1) % 2 == 0
        return -self.config.contempt if leaf_is_root_side else self.config.contempt
