"""Teacher distillation of a stronger position value for the same chess agent.

The teacher (Stockfish through the teaching environment) labels positions with a
shallow search score and its preferred move. The agent learns a small neural
value over public board tokens plus the attack field produced by its *own*
learned rule model. No piece value, square table or tactical rule is authored.
Inference is NumPy only; the teacher is never consulted while playing.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from pai.chess.representation import BoardObservation
from pai.chess.rules import learned_attack_fields

CP_CLIP = 1500.0
FEATURES = 12 * 64 + 2 * 64 + 4  # token planes, own/opponent attack counts, castling


def canonical(boards, turns, castling):
    """Side-to-move perspective: flip ranks and swap colours when Black moves."""
    boards = np.asarray(boards, dtype=np.int16).reshape(-1, 64)
    turns = np.asarray(turns).reshape(-1)
    castling = np.asarray(castling, dtype=np.float32).reshape(-1, 4)
    out = boards.copy()
    black = turns == 0
    if black.any():
        flipped = boards[black].reshape(-1, 8, 8)[:, ::-1, :].reshape(-1, 64)
        swapped = np.where(flipped == 0, 0, np.where(flipped > 6, flipped - 6, flipped + 6))
        out[black] = swapped
        castling = castling.copy()
        castling[black] = castling[black][:, [2, 3, 0, 1]]
    return out, castling


def features_from_arrays(boards, turns, castling, attack_table):
    own_boards, own_castling = canonical(boards, turns, castling)
    n = len(own_boards)
    planes = np.zeros((n, 12, 64), dtype=np.float32)
    rows, squares = np.nonzero(own_boards)
    planes[rows, own_boards[rows, squares] - 1, squares] = 1.0
    _, white, black = learned_attack_fields(own_boards, attack_table)
    # After canonicalisation "white" tokens always belong to the side to move.
    return np.concatenate((planes.reshape(n, -1), white.astype(np.float32),
                           black.astype(np.float32), own_castling), axis=1)


def features(observations, attack_table):
    obs = list(observations)
    return features_from_arrays([o.pieces for o in obs], [o.turn for o in obs],
                                [o.castling for o in obs], attack_table)


def cp_to_prob(cp):
    return 1.0 / (1.0 + np.power(10.0, -np.clip(cp, -CP_CLIP, CP_CLIP) / 400.0))


def prob_to_cp(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.clip(400.0 * np.log10(p / (1 - p)), -CP_CLIP, CP_CLIP)


class DistilledValue:
    """MLP value, side-to-move internally, reported from White's perspective (cp)."""

    def __init__(self, layers, attack_table, summary=None):
        self.layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in layers]
        self.attack_table = np.asarray(attack_table, dtype=np.int8)
        self.summary = summary or {}
        if not self.layers or self.layers[0][0].shape[0] != FEATURES or self.layers[-1][0].shape[1] != 1:
            raise ValueError("unexpected distilled value shape")
        for i, (w, b) in enumerate(self.layers):
            if w.ndim != 2 or b.shape != (w.shape[1],) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError("finite matching distilled layers required")
            if i and self.layers[i - 1][0].shape[1] != w.shape[0]:
                raise ValueError("incompatible distilled layers")
        if self.attack_table.shape != (13, 64, 64, 2):
            raise ValueError("learned attack field required")

    def stm_logits(self, x):
        h = x
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    def predict_many(self, observations):
        observations = list(observations)
        if not observations:
            return []
        x = features(observations, self.attack_table)
        cp = np.clip(self.stm_logits(x) * (400 / np.log(10)), -CP_CLIP, CP_CLIP)
        sign = np.array([1.0 if o.turn else -1.0 for o in observations])
        return (cp * sign).tolist()

    def predict(self, observation):
        return float(self.predict_many([observation])[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"attack_table": self.attack_table}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        arrays["summary"] = np.frombuffer(json.dumps(self.summary).encode(), dtype=np.uint8)
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        count = sum(1 for k in data.files if k.startswith("w"))
        layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(count)]
        summary = json.loads(bytes(data["summary"]).decode())
        return cls(layers, data["attack_table"], summary)


# --------------------------------------------------------------------------- teacher data

def position_hash(fen):
    """Mirror-orbit identity of public positions, excluding move counters."""
    import chess
    from pai.chess.teacher import observation, position_key
    board = chess.Board(fen)
    return min(position_key(observation(board)), position_key(observation(board.mirror())))

def register_rows(value, rows):
    import chess
    from pai.chess.teacher import observation, position_key
    keys, orbits = set(), set()
    for row in rows:
        board = chess.Board(row[0])
        keys.update((position_key(observation(board)), position_key(observation(board.mirror()))))
        orbits.add(position_hash(row[0]))
    value.summary.update(training_positions=sorted(keys), training_position_hashes=sorted(orbits))
    return value


def generate_teacher_positions(args):
    """Worker: teacher-labelled positions from mixed-strength, partly random games.

    args = (engine_path, seed, count, depth). Returns rows of
    (fen, stm_cp, best_uci). Labels come from the teacher only.
    """
    import chess
    import chess.engine
    engine_path, seed, count, depth = args
    rng = random.Random(seed)
    rows = []
    game_id = 0
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        while len(rows) < count:
            game_id += 1
            board = chess.Board()
            for _ in range(rng.randrange(0, 9)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            # Mixed-quality play: mostly teacher-preferred moves, plausible
            # alternatives and occasional uniform blunders on both sides.
            epsilon = rng.choice((0.05, 0.1, 0.2, 0.3))
            for _ in range(rng.randrange(30, 200)):
                if board.is_game_over(claim_draw=True) or len(rows) >= count:
                    break
                infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=4)
                score = infos[0]["score"].pov(board.turn)
                cp = float(np.clip(score.score(mate_score=100000), -CP_CLIP, CP_CLIP))
                best = infos[0].get("pv", [None])[0]
                rows.append((board.fen(), cp, None if best is None else best.uci(), f"{seed}:{game_id}"))
                moves = list(board.legal_moves)
                draw = rng.random()
                if best is None or draw < epsilon * 0.4:
                    move = rng.choice(moves)
                elif draw < epsilon:
                    move = rng.choice([i["pv"][0] for i in infos if i.get("pv")])
                else:
                    move = best
                board.push(move)
    return rows


def rows_to_arrays(rows, attack_table, batch=4096):
    import chess
    from pai.chess.teacher import observation
    xs, ys = [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        obs = [observation(chess.Board(row[0])) for row in chunk]
        xs.append(features(obs, attack_table).astype(np.uint8))
        ys.append(np.array([row[1] for row in chunk], dtype=np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def train_value(x, y_cp, x_val, y_val_cp, hidden=(256, 64), epochs=30, seed=0, lr=1e-3,
                batch=1024, device="cpu", progress=print):
    """Win-probability cross-entropy on teacher scores; selection by validation MAE."""
    import torch
    torch.manual_seed(seed)
    dims = (x.shape[1],) + tuple(hidden) + (1,)
    modules = []
    for i in range(len(dims) - 1):
        modules.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modules.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*modules).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(cp_to_prob(y_cp).astype(np.float32), device=device)
    xv = torch.tensor(x_val, device=device)
    best, best_state, history = math.inf, None, []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(xt), generator=gen).to(device)
        total = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            logits = net(xt[idx])[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(xv)[:, 0]).cpu().numpy()
        mae = float(np.mean(np.abs(prob_to_cp(pv) - y_val_cp)))
        history.append({"epoch": epoch, "train_bce": total / len(xt), "validation_mae_cp": mae})
        progress(f"epoch {epoch}: bce {total / len(xt):.4f} val MAE {mae:.1f} cp", flush=True)
        if mae < best:
            best = mae
            best_state = [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
                          for m in net if isinstance(m, torch.nn.Linear)]
    return best_state, history


def evaluate_rows(value, rows, attack_table=None):
    """MAE (cp, side-to-move frame, clipped) and win-probability error on rows."""
    import chess
    from pai.chess.teacher import observation
    if not rows:
        raise ValueError("nonempty value evaluation required")
    obs = [observation(chess.Board(row[0])) for row in rows]
    actual = np.array([row[1] for row in rows])
    raw = value.predict_many(obs) if hasattr(value, "predict_many") else [value.predict(o) for o in obs]
    predicted = np.array(raw) * np.array([1 if o.turn else -1 for o in obs])
    predicted = np.clip(predicted, -CP_CLIP, CP_CLIP)
    mask = np.abs(actual) >= 100
    return {"positions": len(rows), "mae_cp": float(np.mean(np.abs(predicted - actual))),
            "rmse_cp": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "win_prob_mae": float(np.mean(np.abs(cp_to_prob(predicted) - cp_to_prob(actual)))),
            "zero_baseline_mae_cp": float(np.mean(np.abs(actual))),
            "advantage_sign_accuracy": float(np.mean((predicted[mask] > 0) == (actual[mask] > 0))) if mask.any() else None,
            "advantage_sign_examples": int(mask.sum())}


def observation_from_fen(fen):
    import chess
    from pai.chess.teacher import observation
    return observation(chess.Board(fen))


__all__ = ["DistilledValue", "features", "generate_teacher_positions", "train_value", "evaluate_rows",
           "position_hash", "rows_to_arrays", "BoardObservation"]
