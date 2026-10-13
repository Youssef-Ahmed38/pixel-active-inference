"""Bounded lookahead over learned states, legality, endings and position value."""
from dataclasses import dataclass
import time

from pai.chess.representation import BoardAction
from pai.chess.state import GameState
from pai.cognition.contracts import SkillCommand


@dataclass(frozen=True)
class SearchResult:
    action: BoardAction | None
    claim_draw: bool
    score_cp: float | None
    completed_depth: int
    nodes: int
    budget_exhausted: bool
    elapsed_seconds: float
    principal_variation: tuple[BoardAction, ...] = ()

    def command(self):
        if self.claim_draw:
            return SkillCommand("claim_draw", arguments=() if self.action is None else (self.action.uci(),))
        return None if self.action is None else self.action.command()


class _BudgetExceeded(Exception):
    pass


class LearnedChessSearch:
    """Iterative-deepening negamax with full-width alpha-beta pruning.

    An optional draw claim is an additional zero-valued action. It never removes
    playable moves. Only complete iterations replace the previously chosen move.
    """
    def __init__(self, state_model, evaluator, value, ordering_value=None):
        if evaluator.state_model is not state_model:
            raise ValueError("search and evaluator must share one state model")
        self.state_model = state_model
        self.evaluator = evaluator
        self.value = value
        # A candidate leaf evaluator can be tested without also changing the
        # traversal order and therefore the depth completed under a node cap.
        self.ordering_value = value if ordering_value is None else ordering_value
        self._prediction_cache = {}

    def _predictions(self, model, observations):
        missing = [o for o in observations if (id(model), o) not in self._prediction_cache]
        if missing:
            scores = model.predict_many(missing) if hasattr(model, "predict_many") else [model.predict(o) for o in missing]
            self._prediction_cache.update(((id(model), o), float(score)) for o, score in zip(missing, scores))
        return [self._prediction_cache[(id(model), o)] for o in observations]

    def _children(self, state, actions):
        # Templates contain learned effects with counters rooted at 0/1.
        templates = self.evaluator.successor_templates(state.board)
        all_actions = self.evaluator.legal_actions(state.board)
        by_action = dict(zip(all_actions, templates))
        history = state.history or (state.board,)
        children = []
        for action in actions:
            template = by_action[action]
            half = 0 if template.halfmove_clock == 0 else state.halfmove_clock + template.halfmove_clock
            child = GameState(template.board, half, state.fullmove_number + template.fullmove_number - 1,
                              history + (template.board,))
            children.append((action, child))
        observations = [child.board for _, child in children]
        scores = self._predictions(self.ordering_value, observations)
        children = [(score * (1 if state.board.turn else -1), action, child)
                    for score, (action, child) in zip(scores, children)]
        return sorted(children, key=lambda row: (-row[0], row[1].uci()))

    def claim_option(self, state, actions=None):
        actions = self.evaluator.legal_actions(state.board) if actions is None else tuple(actions)
        automatic = self.evaluator.status(state, actions=actions)
        if automatic.terminal:
            return False, None
        if self.evaluator.current_claim(state, actions=actions) is not None:
            return True, None
        claimed = self.evaluator.status(state, claim_draw=True, actions=actions)
        if not claimed.terminal:
            return False, None
        for _, action, child in self._children(state, actions):
            reason = self.evaluator.current_claim(child)
            if reason == claimed.termination:
                return True, action
        # A prospective claim without a valid witness must not become a command.
        return False, None

    def search(self, state, max_depth=3, max_nodes=256, time_limit=None, max_quiescence=0, reuse_pv=False):
        if max_depth < 1 or max_nodes < 1 or max_quiescence < 0 or (time_limit is not None and time_limit <= 0):
            raise ValueError("depth, node budget and optional time limit must be positive")
        # perf_counter uses the high-resolution platform clock. On Windows,
        # monotonic can have a 15 ms tick and miss short search deadlines.
        start = time.perf_counter()
        # Only board-value calls are cached, never history-dependent search
        # scores. A fresh cache also permits deliberate model changes between moves.
        self._prediction_cache = {}
        deadline = float("inf") if time_limit is None else start + time_limit
        nodes = 0
        exhausted = False
        preferred = {}

        def ordered(position, actions, path):
            children = self._children(position, actions)
            action = preferred.get(path)
            if action is not None:
                children.sort(key=lambda row: row[1] != action)
            return children

        def visit(position, depth, alpha, beta, ply, quiet_left, path):
            nonlocal nodes
            if nodes >= max_nodes or time.perf_counter() >= deadline:
                raise _BudgetExceeded
            nodes += 1
            actions = self.evaluator.legal_actions(position.board)
            status = self.evaluator.status(position, actions=actions)
            if status.terminal:
                score = 0.0 if status.winner is None else (100000.0 - ply) * (1 if status.winner == position.board.turn else -1)
                return score, (), False, None
            claim, witness = self.claim_option(position, actions)
            if depth == 0:
                score = self._predictions(self.value, [position.board])[0] * (1 if position.board.turn else -1)
                if quiet_left == 0:
                    return (max(0.0, score) if claim else score), (), claim and score <= 0, witness
                safe = self.evaluator.safe(position.board)
                best = (max(0.0, score) if claim else score) if safe else (0.0 if claim else -float("inf"))
                best_pv, best_claim = (), claim and (not safe or score <= 0)
                alpha = max(alpha, best)
                if alpha >= beta:
                    return best, best_pv, best_claim, witness
                occupied = sum(piece != 0 for piece in position.board.pieces)
                for _, action, child in ordered(position, actions, path):
                    # Generic learned state effect: fewer occupied cells. The
                    # world model determines which actions have that effect.
                    if safe and sum(piece != 0 for piece in child.board.pieces) >= occupied:
                        continue
                    child_score, pv, _, _ = visit(child, 0, -beta, -alpha, ply + 1, quiet_left - 1, path + (action,))
                    child_score = -child_score
                    if child_score > best:
                        best, best_pv, best_claim = child_score, (action,) + pv, False
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break
                return best, best_pv, best_claim, witness if best_claim else None
            best = 0.0 if claim else -float("inf")
            best_pv, best_claim, best_witness = (), claim, witness
            alpha = max(alpha, best)
            if alpha >= beta:
                return best, best_pv, best_claim, best_witness
            for _, action, child in ordered(position, actions, path):
                score, pv, _, _ = visit(child, depth - 1, -beta, -alpha, ply + 1, quiet_left, path + (action,))
                score = -score
                if score > best:
                    best, best_pv, best_claim, best_witness = score, (action,) + pv, False, None
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best, best_pv, best_claim, best_witness

        # Root setup supplies a safe legal fallback even if an iteration runs out
        # of budget. This is bounded state search, not a hard-real-time system.
        actions = self.evaluator.legal_actions(state.board)
        status = self.evaluator.status(state, actions=actions)
        if status.terminal:
            score = 0.0 if status.winner is None else 100000.0 * (1 if status.winner == state.board.turn else -1)
            return SearchResult(None, False, score, 0, 0, False, time.perf_counter() - start)
        fallback = self._children(state, actions)[0][1] if actions else None
        claim, witness = self.claim_option(state, actions)
        score, pv, selected_claim, selected_witness = (0.0 if claim else None), (() if fallback is None else (fallback,)), claim, witness
        completed = 0
        for depth in range(1, max_depth + 1):
            try:
                outcome = visit(state, depth, -float("inf"), float("inf"), 0, max_quiescence, ())
            except _BudgetExceeded:
                exhausted = True
                break
            score, pv, selected_claim, selected_witness = outcome
            completed = depth
            if reuse_pv:
                # Prefixes identify the exact path from this root, including its
                # history. This stores ordering hints, never cached scores.
                preferred = {pv[:i]: action for i, action in enumerate(pv)}
        action = selected_witness if selected_claim else (pv[0] if pv else fallback)
        return SearchResult(action, selected_claim, None if score is None else float(score), completed, nodes, exhausted,
                            time.perf_counter() - start, pv)

    def command(self, belief, **budgets):
        if ("game_over", "board", None) in belief.relations:
            return None
        return self.search(GameState.from_belief(belief), **budgets).command()
