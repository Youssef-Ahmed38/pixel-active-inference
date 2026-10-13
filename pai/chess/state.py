"""Complete public game state and learned transitions, without a chess oracle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pai.chess.representation import BoardAction, BoardObservation
from pai.chess.rules import LearnedChessRules, LearnedTree
from pai.cognition.contracts import EntityState, StateBelief


@dataclass(frozen=True)
class GameState:
    board: BoardObservation
    halfmove_clock: int = 0
    fullmove_number: int = 1
    # Public positions since the supplied root, including the current position.
    # An imported FEN alone cannot reconstruct earlier repetition history.
    history: tuple[BoardObservation, ...] = ()

    def __post_init__(self):
        if self.halfmove_clock < 0 or self.fullmove_number < 1:
            raise ValueError("invalid game counters")
        if self.history and self.history[-1] != self.board:
            raise ValueError("history must end at the current board")

    def to_belief(self, step=0):
        entities = [EntityState(f"square_{i}", "square", {"index": i, "piece": p})
                    for i, p in enumerate(self.board.pieces)]
        features = {"turn": self.board.turn, "ep_square": self.board.ep_square,
                    "halfmove_clock": self.halfmove_clock, "fullmove_number": self.fullmove_number,
                    "position_history": json.dumps([[o.pieces, o.turn, o.castling, o.ep_square]
                                                    for o in self.history])}
        features.update({f"castle_{i}": flag for i, flag in enumerate(self.board.castling)})
        entities.append(EntityState("board", "board", features))
        return StateBelief("chess", step, tuple(entities), frozenset())

    @classmethod
    def from_belief(cls, belief):
        board = BoardObservation.from_belief(belief)
        features = next(e.features for e in belief.entities if e.kind == "board")
        history = tuple(BoardObservation(tuple(pieces), turn, tuple(rights), ep)
                        for pieces, turn, rights, ep in json.loads(features.get("position_history", "[]")))
        return cls(board, int(features.get("halfmove_clock", 0)),
                   int(features.get("fullmove_number", 1)), history)


def transition_features(obs, actions):
    """Token/square identities and geometry; no piece-specific rule predicates."""
    a = np.asarray(actions, dtype=int).reshape(-1, 3)
    src, dst, promo = a.T
    tokens = np.asarray(obs.pieces)
    n = len(a)
    return np.column_stack((tokens[src], tokens[dst], np.full(n, obs.turn),
                            dst % 8 - src % 8, dst // 8 - src // 8, promo,
                            np.eye(13)[tokens[src]], np.eye(13)[tokens[dst]],
                            np.eye(64)[src], np.eye(64)[dst],
                            np.tile(obs.castling, (n, 1)))).astype(np.float32)


class LearnedChessStateModel:
    """Compose learned piece effects with independently taught metadata updates."""

    def __init__(self, rules, trees, training_positions=(), training_summary=None):
        self.rules = rules
        self.trees = trees
        self.training_positions = tuple(training_positions)
        self.training_summary = training_summary or {}

    def successors(self, state, actions=None):
        """Predict successors in a batch. Rejected actions preserve all state."""
        actions = self.rules.legal_actions(state.board) if actions is None else tuple(actions)
        if not actions:
            return ()
        a = np.array([(m.source, m.destination, m.promotion) for m in actions])
        legal = self.rules.classify(state.board, a)
        x = transition_features(state.board, a)
        predicted = {name: tree.predict(x).astype(int) for name, tree in self.trees.items()}
        # Use the same validated batch for piece effects, avoiding repeated legality calls.
        from pai.chess.rules import spatial_features
        pieces = self.rules._effects(state.board, a, spatial_features(state.board, a, self.rules.attack_table))
        result = []
        for i, action in enumerate(actions):
            if not legal[i]:
                result.append(state)
                continue
            offset = predicted["ep_offset"][i]
            ep = 64 if offset == 64 else action.source + offset
            obs = BoardObservation(tuple(map(int, pieces[i])), int(predicted["turn"][i]),
                                   tuple(int(predicted[f"castle_{j}"][i]) for j in range(4)), int(ep))
            halfmove = 0 if predicted["reset_clock"][i] else state.halfmove_clock + 1
            fullmove = state.fullmove_number + int(predicted["fullmove_delta"][i])
            history = (state.history or (state.board,)) + (obs,)
            result.append(GameState(obs, halfmove, fullmove, history))
        return tuple(result)

    def predict(self, state: GameState, action: BoardAction):
        return self.successors(state, (action,))[0]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "trees": {k: v.to_dict() for k, v in self.trees.items()},
                                    "training_positions": self.training_positions,
                                    "training_summary": self.training_summary}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, rules: LearnedChessRules):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported state checkpoint")
        return cls(rules, {k: LearnedTree.from_dict(v) for k, v in data["trees"].items()},
                   data["training_positions"], data["training_summary"])
