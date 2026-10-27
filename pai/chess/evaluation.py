"""Learned terminal/draw evaluation and position value; oracle-free inference."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from pai.chess.representation import BoardAction
from pai.chess.rules import LearnedTree, safety_features
from pai.chess.state import transition_features
from pai.chess.state import GameState


def material_features(obs):
    pieces = np.asarray(obs.pieces)
    parity = (np.arange(64) % 8 + np.arange(64) // 8) % 2
    counts = [np.bincount(pieces[mask], minlength=13)[1:]
              for mask in (np.ones(64, dtype=bool), parity == 0, parity == 1)]
    # Generic color-invariant token evidence reduces accidental color/parity correlations.
    return np.concatenate(counts + [c[:6] + c[6:] for c in counts])


@dataclass(frozen=True)
class GameStatus:
    termination: str | None
    winner: int | None = None  # 1 white, 0 black, None draw/ongoing.

    @property
    def terminal(self):
        return self.termination is not None


class LearnedChessEvaluator:
    def __init__(self, state_model, trees, reasons, training_keys=(), summary=None):
        self.state_model = state_model
        self.trees = trees
        self.reasons = tuple(reasons)
        self.training_keys = tuple(training_keys)
        self.summary = summary or {}

    @lru_cache(maxsize=8192)
    def legal_actions(self, obs):
        return self.state_model.rules.legal_actions(obs)

    @lru_cache(maxsize=8192)
    def successor_templates(self, obs):
        root = GameState(obs)
        # Counters are represented by reset/delta, independent of root values.
        return self.state_model.successors(root, self.legal_actions(obs))

    @lru_cache(maxsize=8192)
    def safe(self, obs):
        rules = self.state_model.rules
        return bool(rules.safety.predict(np.array([safety_features(obs.pieces, obs.turn, rules.attack_table)]))[0])

    @lru_cache(maxsize=8192)
    def repetition_key(self, obs):
        ep = 64
        if obs.ep_square != 64:
            actions = np.array([(s, obs.ep_square, 0) for s in range(64) if s != obs.ep_square])
            x = np.column_stack((transition_features(obs, actions), actions[:, 1] == obs.ep_square))
            relevant = self.trees["ep_move"].predict(x).astype(bool)
            if np.any(relevant) and np.any(self.state_model.rules.classify(obs, actions[relevant])):
                ep = obs.ep_square
        return obs.pieces, obs.turn, obs.castling, ep

    def features(self, state, claim_draw=False, actions=None):
        rules = self.state_model.rules
        actions = self.legal_actions(state.board) if actions is None else tuple(actions)
        counts = Counter(self.repetition_key(o) for o in (state.history or (state.board,)))
        repeated = counts[self.repetition_key(state.board)]
        safe = int(self.safe(state.board))
        insufficient = int(self.trees["material"].predict(material_features(state.board)[None])[0])
        next_repeated = next_clock = 0
        gate = self.trees["lookahead"].predict(np.array([[state.halfmove_clock, max(counts.values())]]))[0]
        if claim_draw and actions and gate:
            for successor in self.successor_templates(state.board):
                next_repeated = max(next_repeated, counts[self.repetition_key(successor.board)] + 1)
                clock = 0 if successor.halfmove_clock == 0 else state.halfmove_clock + successor.halfmove_clock
                if clock > next_clock and self.legal_actions(successor.board):
                    next_clock = clock
        return np.array([[len(actions), safe, insufficient, state.halfmove_clock,
                          repeated, next_repeated, next_clock, state.board.turn, int(claim_draw)]])

    def status(self, state, claim_draw=False, actions=None):
        x = self.features(state, claim_draw, actions)
        reason_id = int(self.trees["outcome"].predict(x)[0])
        reason = self.reasons[reason_id]
        winner = int(self.trees["winner"].predict(np.array([[reason_id, state.board.turn]]))[0])
        return GameStatus(None if reason == "ONGOING" else reason, None if winner < 0 else winner)

    def current_claim(self, state, actions=None):
        """Eligibility in this position, separately taught from prospective claims."""
        if "current_claim" not in self.trees:
            raise ValueError("ending checkpoint needs current-claim supervision; retrain chess_evaluation.py")
        features = self.features(state, actions=actions)
        reason = self.reasons[int(self.trees["current_claim"].predict(features)[0])]
        return None if reason == "ONGOING" else reason

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "trees": {k: t.to_dict() for k, t in self.trees.items()},
                                    "reasons": self.reasons, "training_keys": self.training_keys,
                                    "summary": self.summary}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, state_model):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported evaluator checkpoint")
        return cls(state_model, {k: LearnedTree.from_dict(v) for k, v in data["trees"].items()},
                   data["reasons"], data["training_keys"], data["summary"])


def value_features(obs):
    """Signed token-square and token-count evidence, canonical across colors."""
    pieces = np.array(obs.pieces).reshape(8, 8)
    codes = np.arange(1, 7)[:, None, None]
    x = ((pieces[None] == codes).astype(np.float32) - (pieces[None, ::-1] == codes + 6).astype(np.float32)).reshape(6,64)
    return np.concatenate((x.sum(1), x.reshape(-1), [2 * obs.turn - 1])).astype(np.float32)


class LearnedPositionValue:
    """Teacher-distilled centipawn value from White's perspective."""
    def __init__(self, weights, summary=None):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.summary = summary or {}
        if self.weights.shape != (391,):
            raise ValueError("unexpected value feature shape")

    def predict(self, obs):
        return float(value_features(obs) @ self.weights)

    def predict_many(self, observations):
        observations = list(observations)
        if not observations:
            return np.empty(0, dtype=float)
        boards = np.asarray([o.pieces for o in observations]).reshape(-1, 8, 8)
        codes = np.arange(1, 7)[None, :, None, None]
        signed = ((boards[:, None] == codes).astype(np.float32) -
                  (boards[:, None, ::-1] == codes + 6).astype(np.float32)).reshape(-1, 6, 64)
        x = np.concatenate((signed.sum(2), signed.reshape(-1, 384),
                            np.asarray([[2 * o.turn - 1] for o in observations],dtype=np.float32)), axis=1)
        return x @ self.weights

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "weights": self.weights.tolist(), "summary": self.summary}) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported value checkpoint")
        return cls(data["weights"], data["summary"])

    def choose(self, state_model, evaluator, state):
        actions = state_model.rules.legal_actions(state.board)
        if evaluator.status(state, actions=actions).terminal:
            return None
        states = state_model.successors(state, actions)
        scores = []
        for successor in states:
            status = evaluator.status(successor)
            score = self.predict(successor.board)
            if status.terminal:
                score = 0.0 if status.winner is None else (100000.0 if status.winner else -100000.0)
            scores.append(score if state.board.turn else -score)
        return actions[int(np.argmax(scores))] if actions else None

    def command(self, state_model, evaluator, belief):
        if ("game_over", "board", None) in belief.relations:
            return None
        action = self.choose(state_model, evaluator, GameState.from_belief(belief))
        return None if action is None else action.command()
