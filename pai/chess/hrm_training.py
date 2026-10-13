"""Teacher-side fitting and validation of chess HRM value residuals."""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from pai.chess.hrm_value import HRMPositionValue, board_tensors, mirror_observation
from pai.chess.teacher import position_key
from pai.thinker.hrm import HRM


def orbit_key(observation):
    return min(position_key(observation), position_key(mirror_observation(observation)))


def partition(rows):
    train, validation = [], []
    for row in rows:
        (validation if int(orbit_key(row[0])[:8], 16) % 5 == 0 else train).append(row)
    if not train or not validation:
        raise ValueError("teaching requires nonempty disjoint train and validation sets")
    return train, validation


def prediction_metrics(value, rows, segments=None):
    if not rows:
        raise ValueError("nonempty evaluation required")
    predicted = []
    for start in range(0, len(rows), 64):
        observations = [o for o, _, _ in rows[start:start + 64]]
        predicted.extend(value.predict_many(observations, segments) if isinstance(value, HRMPositionValue)
                         else [value.predict(o) for o in observations])
    y, predicted = np.array([y for _, y, _ in rows]), np.array(predicted)
    return {"positions": len(rows), "mae_cp": float(np.mean(np.abs(y - predicted))),
            "rmse_cp": float(np.sqrt(np.mean((y - predicted) ** 2)))}


def fit_hrm_value(baseline, rows, epochs=16, seed=700, batch_size=32, progress=print):
    if epochs < 1 or batch_size < 1:
        raise ValueError("positive training budgets required")
    train, validation = partition(rows)
    torch.manual_seed(seed)
    config = {"in_dim": 16, "ctx_dim": 5, "max_len": 64, "heads": {"value": 1},
              "d_model": 32, "n_heads": 4, "h_layers": 1, "l_layers": 1,
              "h_cycles": 1, "l_steps": 2, "max_segments": 2}
    model = HRM(**config)
    # The initial residual is exactly zero, preserving the established baseline.
    torch.nn.init.zeros_(model.readout.heads["value"].weight)
    torch.nn.init.zeros_(model.readout.heads["value"].bias)
    value = HRMPositionValue(model, baseline, config)
    x, mask, ctx = board_tensors([o for o, _, _ in train] + [mirror_observation(o) for o, _, _ in train])
    targets = torch.tensor([(score - baseline.predict(o)) / value.residual_scale for o, score, _ in train], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    best_metric = prediction_metrics(baseline, validation)["mae_cp"]
    best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), batch_size):
            ix = order[start:start + batch_size]
            paired = torch.cat((ix, ix + len(train)))
            outputs = model.forward_segments(x[paired], mask[paired], ctx[paired])
            losses = []
            for output in outputs:
                raw = output.logits["value"].squeeze(-1)
                prediction = (raw[:len(ix)] - raw[len(ix):]) / 2
                losses.append(F.smooth_l1_loss(prediction, targets[ix]))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        model.eval()
        metric = prediction_metrics(value, validation)["mae_cp"]
        if metric < best_metric:
            best_metric, best_state, best_epoch = metric, copy.deepcopy(model.state_dict()), epoch + 1
        row = {"epoch": epoch + 1, "training_loss": total / len(train), "validation_mae_cp": metric}
        history.append(row)
        progress(f"HRM epoch {epoch + 1}/{epochs}: validation MAE {metric:.1f} cp, best epoch {best_epoch}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    keys = set(baseline.summary.get("training_positions", ()))
    keys.update(position_key(o) for o, _, _ in rows)
    keys.update(position_key(mirror_observation(o)) for o, _, _ in rows)
    value.summary = {"architecture": "existing pai.thinker.hrm.HRM, scalar chess residual",
                     "training_examples": len(train), "validation_examples": len(validation),
                     "selected_epoch": best_epoch, "validation_mae_cp": best_metric,
                     "training_positions": sorted(keys), "fixed_segments": 2,
                     "learned_halting": False}
    return value, history
