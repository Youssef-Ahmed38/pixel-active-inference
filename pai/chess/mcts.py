"""UCT tree search over learned chess states, with optional claim actions.

No teacher, chess library, transposition merge or authored move rules are used.
This branch has no HRM yet and is not the incumbent playing policy.
"""
from dataclasses import dataclass, field
import math
import time

from pai.chess.search import LearnedChessSearch, SearchResult


@dataclass(frozen=True)
class MCTSResult(SearchResult):
    simulations: int = 0
    deepest_ply: int = 0
    selected_visits: int = 0


@dataclass
class _Node:
    state: object
    action: object = None
    claim: bool = False
    visits: int = 0
    total: float = 0
    children: list = field(default_factory=list)
    pending: list | None = None
    leaf_value: float = 0
    terminal: bool = False


class LearnedChessMCTS(LearnedChessSearch):
    def search(self, state, max_depth=3, max_nodes=256, time_limit=None, exploration=1.4):
        if max_depth < 1 or max_nodes < 1 or exploration < 0 or (time_limit is not None and time_limit <= 0):
            raise ValueError("positive budgets and nonnegative exploration required")
        start = time.perf_counter()
        deadline = float("inf") if time_limit is None else start + time_limit
        count = deepest = simulations = 0

        def prepare(node):
            if node.claim:
                node.terminal, node.leaf_value, node.pending = True, 0.0, []
                return
            actions = self.evaluator.legal_actions(node.state.board)
            status = self.evaluator.status(node.state, actions=actions)
            if status.terminal:
                node.terminal = True
                node.leaf_value = (0.0 if status.winner is None else
                                   1.0 if status.winner == node.state.board.turn else -1.0)
                node.pending = []
                return
            claim, witness = self.claim_option(node.state, actions)
            score = self.value.predict(node.state.board) * (1 if node.state.board.turn else -1)
            node.leaf_value = math.tanh(score / 400)
            if claim:
                node.leaf_value = max(0.0, node.leaf_value)
            node.pending = [_Node(child, action) for _, action, child in self._children(node.state, actions)]
            if claim:
                node.pending.insert(0, _Node(None, witness, True))

        root = _Node(state)
        prepare(root)
        if root.terminal:
            return MCTSResult(None, False, root.leaf_value * 100000, 0, 0, False,
                              time.perf_counter() - start)
        fallback = root.pending[0]
        while count < max_nodes and time.perf_counter() < deadline:
            node, path, depth = root, [root], 0
            count += 1
            while not node.terminal and depth < max_depth:
                if count >= max_nodes or time.perf_counter() >= deadline:
                    break
                if node.pending:
                    child = node.pending.pop(0)
                    node.children.append(child)
                    prepare(child)
                    node = child
                    path.append(node)
                    depth += 1
                    count += 1
                    break
                if not node.children:
                    break
                def ucb(child):
                    mean = -child.total / child.visits if child.visits else 0.0
                    return (mean + exploration * math.sqrt(math.log(max(2, node.visits)) / max(1, child.visits)))
                node = max(node.children, key=ucb)
                path.append(node)
                depth += 1
                count += 1
            deepest = max(deepest, depth)
            value = node.leaf_value
            for visited in reversed(path):
                visited.visits += 1
                visited.total += value
                value = -value
            simulations += 1
        # Visits are the robust choice; value breaks ties among equally explored moves.
        selected = max(root.children, key=lambda child: (child.visits, -child.total / child.visits)) if root.children else fallback
        score = None
        if selected.visits:
            mean = -selected.total / selected.visits
            score = 400 * math.atanh(max(-.999999, min(.999999, mean)))
        elif selected.claim:
            score = 0.0
        pv = () if selected.claim or selected.action is None else (selected.action,)
        return MCTSResult(selected.action, selected.claim, score, 0, count,
                          count >= max_nodes or time.perf_counter() >= deadline,
                          time.perf_counter() - start, pv, simulations, deepest, selected.visits)
