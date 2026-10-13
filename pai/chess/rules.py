"""Learned relational rules. Runtime needs NumPy and public observations only.

Geometry and generic token edits are represented here, but chess movement,
attack, special-edit and safety decisions are fitted teacher data, not code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pai.chess.representation import BoardAction, BoardObservation, candidate_actions


def _geometry():
    between = np.zeros((64, 64, 64), dtype=np.int8)
    for a in range(64):
        for b in range(64):
            dx, dy = b % 8 - a % 8, b // 8 - a // 8
            if a == b or not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
                continue
            sx, sy = int(np.sign(dx)), int(np.sign(dy))
            x, y = a % 8 + sx, a // 8 + sy
            while (x, y) != (b % 8, b // 8):
                between[a, b, x + 8 * y] = 1
                x, y = x + sx, y + sy
    return between


BETWEEN = _geometry()
PATHS = np.full((64, 64, 6), -1, dtype=np.int16)
for _src in range(64):
    for _dst in range(64):
        _cells = np.flatnonzero(BETWEEN[_src, _dst])
        PATHS[_src, _dst, :len(_cells)] = _cells
try:
    from pai.chess import fast_rules as _compiled
except ImportError:
    _compiled = None
ACTIONS = candidate_actions()
ACTION_ARRAY = np.array([(a.source, a.destination, a.promotion) for a in ACTIONS], dtype=np.int16)


@dataclass
class LearnedTree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    @classmethod
    def from_classifier(cls, model):
        tree = model.tree_
        values = model.classes_[tree.value[:, 0].argmax(1)]
        return cls(tree.feature, tree.threshold, tree.children_left, tree.children_right, values)

    def predict(self, x):
        if _compiled is not None:
            return _compiled.tree_predict(np.asarray(x), self.feature, self.threshold, self.left, self.right, self.value)
        nodes = np.zeros(len(x), dtype=np.int32)
        while True:
            active = np.flatnonzero(self.left[nodes] >= 0)
            if not len(active):
                return self.value[nodes]
            n = nodes[active]
            nodes[active] = np.where(x[active, self.feature[n]] <= self.threshold[n], self.left[n], self.right[n])

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("feature", "threshold", "left", "right", "value")}

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: np.asarray(v) for k, v in data.items()})


@dataclass
class CategoricalTrees:
    """Condition generic decision trees on a categorical observation token."""
    column: int
    groups: dict[int, LearnedTree]
    relation_column: int | None = None
    guards: dict[int, list] | None = None

    @property
    def feature(self):
        return np.concatenate([tree.feature for tree in self.groups.values()])

    def packed(self):
        signature = tuple((code,id(tree),tuple(map(tuple,(self.guards or {}).get(code,())))) for code,tree in self.groups.items())
        if getattr(self,"_packed_signature",None) != signature:
            size = max(self.groups,default=0) + 1
            starts = np.full(size,-1,dtype=np.int64)
            enabled = np.zeros(size,dtype=bool)
            width = max([len(g) for g in (self.guards or {}).values()] + [1])
            cols = np.full((size,width),-1,dtype=np.int64)
            vals = np.zeros((size,width),dtype=np.float64)
            parts = {k: [] for k in ("feature","threshold","left","right","value")}
            offset = 0
            for code,tree in self.groups.items():
                starts[code], enabled[code] = offset, np.any(tree.value)
                for k in parts:
                    data = getattr(tree,k)
                    parts[k].append(np.where(data >= 0,data + offset,data) if k in ("left","right") else data)
                for j,(col,val) in enumerate((self.guards or {}).get(code,())):
                    cols[code,j],vals[code,j] = col,val
                offset += len(tree.feature)
            arrays = [np.concatenate(parts[k]) for k in parts]
            arrays[-1] = arrays[-1].astype(bool)
            self._packed_signature = signature
            self._packed_data = (starts,*arrays,cols,vals,enabled)
        return self._packed_data

    def predict(self, x):
        if _compiled is not None and self.groups:
            packed = self.packed()
            return _compiled.categorical(x,self.column,-1 if self.relation_column is None else self.relation_column,*packed[:-1])
        result = np.zeros(len(x), dtype=bool)
        categories = x[:, self.column].astype(int)
        if self.relation_column is not None:
            categories = categories * 2 + x[:, self.relation_column].astype(int)
        for code, tree in self.groups.items():
            selected = np.flatnonzero(categories == code)
            if len(selected):
                result[selected] = tree.predict(x[selected])
                for column, value in (self.guards or {}).get(code, []):
                    result[selected] &= x[selected, column] == value
        return result

    def to_dict(self):
        return {"column": self.column, "relation_column": self.relation_column,
                "guards": self.guards or {}, "groups": {str(k): v.to_dict() for k, v in self.groups.items()}}

    @classmethod
    def from_dict(cls, data):
        return cls(data["column"], {int(k): LearnedTree.from_dict(v) for k, v in data["groups"].items()},
                   data.get("relation_column"), {int(k): v for k, v in data.get("guards", {}).items()})


def spatial_features_reference(obs, actions, attack_table):
    """Generic coordinates, token identities, paths, and learned attack fields."""
    board = np.array(obs.pieces, dtype=np.int16)
    src, dst, promo = np.asarray(actions).T
    sx, sy, tx, ty = src % 8, src // 8, dst % 8, dst // 8
    dx, dy = tx - sx, ty - sy
    occupied = (board != 0).astype(np.int16)
    blockers = np.tensordot(BETWEEN, occupied, axes=([2], [0]))
    attack = attack_table[board[:, None], np.arange(64)[:, None], np.arange(64)[None, :], (blockers > 0).astype(int)]
    white = ((board > 0) & (board <= 6)).astype(np.int16) @ attack
    black = (board > 6).astype(np.int16) @ attack
    path_white = np.tensordot(BETWEEN, white, axes=([2], [0]))
    path_black = np.tensordot(BETWEEN, black, axes=([2], [0]))
    source_code, target_code = board[src], board[dst]
    base = [source_code, target_code, np.full(len(src), obs.turn), promo,
            sx, sy, tx, ty, dx, dy, np.abs(dx), np.abs(dy),
            blockers[src, dst], (source_code > 6).astype(int), (target_code > 6).astype(int),
            np.full(len(src), obs.ep_square), (dst == obs.ep_square).astype(int),
            white[src], black[src], white[dst], black[dst], path_white[src, dst], path_black[src, dst]]
    base += [np.full(len(src), bit) for bit in obs.castling]
    base += [board[sy * 8 + i] for i in range(8)]
    # Nearby token evidence supports learning additional removal/relocation effects.
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        x, y = tx + ox, ty + oy
        idx = np.clip(x + 8 * y, 0, 63)
        base.append(np.where((x >= 0) & (x < 8) & (y >= 0) & (y < 8), board[idx], -1))
    # Reuse the independently taught relation instead of relearning geometry
    # through absolute-square correlations in the final decision tree.
    base += [attack[src, dst],
             ((source_code > 0) & ((source_code <= 6) == bool(obs.turn))).astype(int),
             ((target_code > 0) & ((target_code <= 6) == bool(obs.turn))).astype(int)]
    other = black if obs.turn else white
    other_path = path_black if obs.turn else path_white
    base += [other[src], other[dst], other_path[src, dst]]
    # Extend the generic movement direction to the workspace boundary. This
    # represents occupancy beyond a proposed endpoint without encoding a rule.
    straight = (dx == 0) | (dy == 0) | (np.abs(dx) == np.abs(dy))
    direction_x, direction_y = np.sign(dx), np.sign(dy)
    distance_x = np.where(direction_x > 0, 7 - sx, np.where(direction_x < 0, sx, 8))
    distance_y = np.where(direction_y > 0, 7 - sy, np.where(direction_y < 0, sy, 8))
    distance = np.minimum(distance_x, distance_y)
    edge = (sx + direction_x * distance) + 8 * (sy + direction_y * distance)
    edge = np.clip(edge, 0, 63)
    base.append(np.where(straight, blockers[src, edge], 0))
    base.append((target_code != 0).astype(int))
    return np.column_stack(base).astype(np.float32)

def spatial_features(obs, actions, attack_table):
    if _compiled is None:
        return spatial_features_reference(obs, actions, attack_table)
    return _compiled.spatial(np.asarray(obs.pieces, dtype=np.int16), obs.turn,
                             np.asarray(obs.castling, dtype=np.int16), obs.ep_square,
                             np.asarray(actions, dtype=np.int16).reshape(-1, 3), PATHS, attack_table)


def safety_features(pieces, turn, attack_table):
    """Attack counts for every token type; the tree learns which must be safe."""
    board = np.asarray(pieces, dtype=np.int16)
    if _compiled is not None:
        return _compiled.safety_batch(board[None],turn,PATHS,attack_table)[0].tolist()
    blockers = np.tensordot(BETWEEN, (board != 0).astype(np.int16), axes=([2], [0]))
    attack = attack_table[board[:, None], np.arange(64)[:, None], np.arange(64)[None, :], (blockers > 0).astype(int)]
    white = ((board > 0) & (board <= 6)).astype(np.int16) @ attack
    black = (board > 6).astype(np.int16) @ attack
    features = [turn]
    for code in range(1, 13):
        positions = np.flatnonzero(board == code)
        features += [int(white[positions].max()) if len(positions) else 0,
                     int(black[positions].max()) if len(positions) else 0]
    return features


BETWEEN_FLAT = BETWEEN.reshape(4096, 64).T.astype(np.float32)  # occupied @ this = blockers


def learned_attack_fields(boards, attack_table):
    """Batched learned attack relation: (B, 64, 64) attack and per-square counts.

    Exactly the same computation as the per-board path in ``spatial_features``
    and ``safety_features``; only the batching differs.
    """
    boards = np.asarray(boards, dtype=np.int16).reshape(-1, 64)
    blocked = ((boards != 0).astype(np.float32) @ BETWEEN_FLAT).reshape(-1, 64, 64) > 0
    squares = np.arange(64)
    attack = np.asarray(attack_table)[boards[:, :, None], squares[None, :, None], squares[None, None, :],
                                      blocked.astype(np.intp)]
    white_mask = ((boards > 0) & (boards <= 6)).astype(np.int16)
    black_mask = (boards > 6).astype(np.int16)
    white = np.einsum("bs,bsd->bd", white_mask, attack.astype(np.int16))
    black = np.einsum("bs,bsd->bd", black_mask, attack.astype(np.int16))
    return attack, white, black


def safety_features_batch(boards, turn, attack_table):
    """Vectorized ``safety_features`` for many boards with the same side to move."""
    boards = np.asarray(boards, dtype=np.int16).reshape(-1, 64)
    if _compiled is not None:
        return _compiled.safety_batch(boards, turn, PATHS, attack_table)
    _, white, black = learned_attack_fields(boards, attack_table)
    columns = [np.full(len(boards), turn)]
    for code in range(1, 13):
        here = boards == code
        # Attack counts are nonnegative, so a zero fill equals the empty-set default.
        columns += [np.where(here, white, 0).max(1), np.where(here, black, 0).max(1)]
    return np.column_stack(columns)


class LearnedChessRules:
    def __init__(self, attack_table, proposal, moved_token, extra_edits, edit_patterns, safety,
                 training_positions=(), training_summary=None):
        self.attack_table = np.asarray(attack_table, dtype=np.int8)
        self.proposal = proposal
        self.moved_token = moved_token
        self.extra_edits = extra_edits
        self.edit_patterns = edit_patterns
        self.safety = safety
        self.training_positions = tuple(training_positions)
        self.training_summary = training_summary or {}

    def _effects(self, obs, actions, features):
        tokens = self.moved_token.predict(features)
        patterns = self.extra_edits.predict(features)
        boards = np.tile(np.asarray(obs.pieces, dtype=np.int16), (len(actions), 1))
        for i, ((src, dst, _), token, pattern) in enumerate(zip(actions, tokens, patterns)):
            boards[i, src] = 0
            boards[i, dst] = token
            for offset, new_token in self.edit_patterns[int(pattern)]:
                index = int(src) + offset
                if 0 <= index < 64:
                    boards[i, index] = new_token
        return boards

    def classify(self, obs, actions=ACTION_ARRAY):
        actions = np.asarray(actions, dtype=np.int16).reshape(-1, 3)
        selected = np.arange(len(actions))
        categorical = isinstance(self.proposal, CategoricalTrees) and self.proposal.column == 0 and self.proposal.relation_column in (None,39,40)
        if categorical and _compiled is not None and self.proposal.groups:
            packed = self.proposal.packed()
            selected = np.flatnonzero(_compiled.prefilter(np.asarray(obs.pieces,dtype=np.int16),obs.turn,actions,
                                     -1 if self.proposal.relation_column is None else self.proposal.relation_column,
                                     packed[-1],packed[-3],packed[-2],PATHS,self.attack_table))
        elif categorical:
            # Algebraic pruning of fitted categorical regions whose leaves are
            # all false (including missing groups). No movement predicate.
            tokens = np.asarray(obs.pieces)[actions[:, 0]]
            own = ((tokens > 0) & ((tokens <= 6) == bool(obs.turn))).astype(int)
            targets = np.asarray(obs.pieces)[actions[:, 1]]
            target_own = ((targets > 0) & ((targets <= 6) == bool(obs.turn))).astype(int)
            cheap = {0: tokens, 3: actions[:, 2], 40: own, 41: target_own, 46: targets != 0}
            if self.proposal.relation_column == 39:
                if _compiled is not None:
                    blockers, attack, _, _ = _compiled.fields(np.asarray(obs.pieces, dtype=np.int16), PATHS, self.attack_table)
                else:
                    blockers = np.tensordot(BETWEEN, np.asarray(obs.pieces) != 0, axes=([2], [0]))
                    attack = self.attack_table[np.asarray(obs.pieces)[:, None], np.arange(64)[:, None], np.arange(64)[None, :], (blockers > 0).astype(int)]
                cheap[39] = attack[actions[:, 0], actions[:, 1]]
                cheap[12] = blockers[actions[:, 0], actions[:, 1]]
            category = tokens if self.proposal.relation_column is None else tokens * 2 + cheap[self.proposal.relation_column]
            eligible = np.zeros(len(actions), dtype=bool)
            for code, tree in self.proposal.groups.items():
                if not np.any(tree.value):
                    continue
                region = category == code
                for col, value in (self.proposal.guards or {}).get(code, ()):
                    if col in cheap:
                        region &= cheap[col] == value
                eligible |= region
            selected = np.flatnonzero(eligible)
        result = np.zeros(len(actions), dtype=bool)
        if not len(selected):
            return result
        actions = actions[selected]
        features = spatial_features(obs, actions, self.attack_table)
        predicted = self.proposal.predict(features).astype(bool)
        proposed = np.flatnonzero(predicted)
        if len(proposed):
            effects = self._effects(obs, actions[proposed], features[proposed])
            safety = safety_features_batch(effects, obs.turn, self.attack_table)
            predicted[proposed] &= self.safety.predict(safety).astype(bool)
        result[selected] = predicted
        return result

    def legal_actions(self, obs):
        return tuple(ACTIONS[i] for i in np.flatnonzero(self.classify(obs)))

    def choose(self, obs):
        """Return a model-approved action, or abstain if none is predicted legal."""
        choices = self.legal_actions(obs)
        return choices[0] if choices else None

    def command(self, belief):
        """Policy bridge to the same cognitive observation/skill interfaces."""
        if ("game_over", "board", None) in belief.relations:
            return None
        action = self.choose(BoardObservation.from_belief(belief))
        return action.command() if action is not None else None

    def predict_piece_board(self, obs, action):
        actions = np.array([(action.source, action.destination, action.promotion)])
        if not self.classify(obs, actions)[0]:
            return obs.pieces
        features = spatial_features(obs, actions, self.attack_table)
        return tuple(map(int, self._effects(obs, actions, features)[0]))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trees = {k: getattr(self, k).to_dict() for k in ("proposal", "moved_token", "extra_edits", "safety")}
        data = {"schema": 1, "attack_table": self.attack_table.tolist(), "trees": trees,
                "edit_patterns": self.edit_patterns, "training_positions": self.training_positions,
                "training_summary": self.training_summary}
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported learned-rule checkpoint")
        trees = {k: (CategoricalTrees.from_dict(v) if "groups" in v else LearnedTree.from_dict(v))
                 for k, v in data["trees"].items()}
        return cls(data["attack_table"], edit_patterns=data["edit_patterns"],
                   training_positions=data["training_positions"], training_summary=data["training_summary"], **trees)
