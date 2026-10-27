"""Stage 2 gates on fresh seeds: stand, walk 10 m, headings, run, speed tracking, pushes, fall causes.

    python scripts/locomotion_eval.py --config configs/locomotion.yaml \
        --policy runs/locomotion/policy/policy.pt --ckpt runs/locomotion/policy/ckpt_last.pt

Seeds: every test uses ``eval.seeds_start + k`` (default 50000..50006, never used in training or
development; training used seed 0, development checks seed 777). The fall-cause calibration
uses seed 60000 (undisturbed falls only). Every body has randomised dynamics (friction, masses,
payload, servo strength) drawn from the eval seed, plus observation noise as in training.
Writes results/locomotion/eval.json (or --name).
"""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts._cli import parse  # noqa: E402
from pai.locomotion.env import EnvConfig, LocomotionEnv, wrap  # noqa: E402
from pai.locomotion.falls import CAUSES, COLLAPSE_STEPS, Calibration, SurpriseMonitor, posterior  # noqa: E402
from pai.locomotion.policy import LocomotionController  # noqa: E402
from pai.locomotion.world_model import EnsembleBodyModel  # noqa: E402

DT = 0.02


def make_env(ecfg, n, seed, episode_s=60.0):
    cfg = copy.deepcopy(ecfg)
    cfg.episode_s = episode_s
    env = LocomotionEnv(n, cfg, seed=seed, stage=len(cfg.stages) - 1)
    env.pushes_enabled = False
    env.set_command(np.arange(n), 0.0, 0.0, 0.0)
    env.reset(np.arange(n))
    return env


class Runner:
    """Steps a batch with the frozen controller; records the first episode of every body."""

    def __init__(self, env, ctrl, monitor=None):
        self.env, self.ctrl, self.monitor = env, ctrl, monitor
        self.obs = env.observe()
        n = env.n
        self.alive = np.ones(n, bool)
        self.fall_t = np.full(n, np.nan)
        self.t = 0.0
        self.start_xy = env.qpos()[:, :2].copy()
        self.start_yaw = env.yaw().copy()
        self.xy = self.start_xy.copy()
        self.fall_features = {}
        self.fall_surprise = {}

    def step(self):
        env = self.env
        a = self.ctrl.act(self.obs)
        nobs, _, done, info = env.step(a)
        self.t += DT
        nxt, sens = nobs.copy(), env.sensors.copy()
        if "done_ids" in info:
            ids = info["done_ids"]
            nxt[ids] = info["final_obs"]
            sens[ids] = info["final_sensors"]
        if self.monitor is not None:
            h, v = env.soles(sens)
            self.monitor.update(self.obs, a, nxt, h, v)
        xy = env.qpos()[:, :2].copy()
        if "done_ids" in info:
            ids = info["done_ids"]
            xy[ids] = info["final_qpos"][:, :2]
            fell = ids[info["fell"][ids]]
            new = fell[self.alive[fell]]
            self.fall_t[new] = self.t
            if self.monitor is not None and len(new):
                f = self.monitor.features(new, exclude_last=COLLAPSE_STEPS)
                s = self.monitor.recent_surprise(new)
                for j, i in enumerate(new):
                    self.fall_features[int(i)] = f[j]
                    self.fall_surprise[int(i)] = float(s[j])
            self.alive[new] = False
        self.xy = np.where(self.alive[:, None], xy, self.xy)
        self.obs = nobs
        return info

    def run(self, seconds, every=None):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if every:
                every(self)

    def forward_distance(self):
        d = self.xy - self.start_xy
        return d[:, 0] * np.cos(self.start_yaw) + d[:, 1] * np.sin(self.start_yaw)


def rate(x):
    return float(np.mean(x))


# ---------------------------------------------------------------- the tests
def test_stand(ecfg, ctrl, seed, n=20, seconds=30.0):
    env = make_env(ecfg, n, seed)
    r = Runner(env, ctrl)
    r.run(seconds)
    drift = np.linalg.norm(r.xy - r.start_xy, axis=1)
    return {"seed": seed, "starts": n, "seconds": seconds, "stood_rate": rate(r.alive),
            "median_drift_m": float(np.median(drift)), "max_drift_m": float(drift.max())}


def test_walk(ecfg, ctrl, seed, n=20, speed=0.8, distance=10.0, min_speed=0.5):
    env = make_env(ecfg, n, seed)
    r0 = Runner(env, ctrl)
    env.set_command(np.arange(n), speed, 0.0, 0.0, heading=r0.start_yaw)
    reach_t = np.full(n, np.nan)

    def watch(r):
        d = r.forward_distance()
        hit = np.isnan(reach_t) & r.alive & (d >= distance)
        reach_t[hit] = r.t

    limit = distance / min_speed
    r0.run(limit, watch)
    ok = ~np.isnan(reach_t)
    d = r0.forward_distance()
    return {"seed": seed, "starts": n, "command_mps": speed, "distance_m": distance,
            "time_limit_s": limit, "success_rate": rate(ok),
            "falls": int((~r0.alive).sum()),
            "mean_speed_to_10m_mps": float(np.nanmean(distance / reach_t)) if ok.any() else None,
            "forward_distance_m": [round(float(x), 2) for x in d]}


def test_headings(ecfg, ctrl, seed, n=20, speed=0.5, seconds=8.0, tol_deg=15.0):
    env = make_env(ecfg, n, seed)
    r = Runner(env, ctrl)
    turns = np.deg2rad(np.array([90, -90, 180, 45, -45, 135, -135, 60, -60, 120] * 10)[:n])
    target = wrap(r.start_yaw + turns)
    env.set_command(np.arange(n), speed, 0.0, 0.0, heading=target)
    r.run(seconds)
    err = np.rad2deg(np.abs(wrap(env.yaw() - target)))
    ok = r.alive & (err < tol_deg)
    return {"seed": seed, "starts": n, "speed_mps": speed, "seconds": seconds,
            "turns_deg": np.rad2deg(turns).round().tolist(),
            "success_rate": rate(ok), "tolerance_deg": tol_deg,
            "median_error_deg": float(np.median(err[r.alive])) if r.alive.any() else None,
            "falls": int((~r.alive).sum())}


def test_speeds(ecfg, ctrl, seed, speeds=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
                per=10, seconds=10.0, measure_s=6.0):
    n = per * len(speeds)
    env = make_env(ecfg, n, seed)
    r = Runner(env, ctrl)
    cmd = np.repeat(speeds, per)
    for i in range(n):
        env.set_command([i], cmd[i], 0.0, 0.0, heading=r.start_yaw[i])
    r.run(seconds - measure_s)
    xy0 = r.xy.copy()
    alive0 = r.alive.copy()
    r.run(measure_s)
    d = r.xy - xy0
    fwd = (d[:, 0] * np.cos(r.start_yaw) + d[:, 1] * np.sin(r.start_yaw)) / measure_s
    rows = []
    for s in speeds:
        m = (cmd == s)
        ok = m & r.alive & alive0
        rows.append({"command_mps": s, "bodies": int(m.sum()), "falls": int((m & ~r.alive).sum()),
                     "mean_forward_mps": float(fwd[ok].mean()) if ok.any() else None,
                     "sd_forward_mps": float(fwd[ok].std()) if ok.any() else None})
    return {"seed": seed, "seconds": seconds, "measured_last_s": measure_s, "rows": rows}


def test_run(ecfg, ctrl, seed, n=20, speed=2.0, seconds=12.0, measure_s=8.0, need=1.5):
    env = make_env(ecfg, n, seed)
    r = Runner(env, ctrl)
    env.set_command(np.arange(n), speed, 0.0, 0.0, heading=r.start_yaw)
    r.run(seconds - measure_s)
    xy0 = r.xy.copy()
    r.run(measure_s)
    d = r.xy - xy0
    fwd = (d[:, 0] * np.cos(r.start_yaw) + d[:, 1] * np.sin(r.start_yaw)) / measure_s
    ok = r.alive & (fwd >= need)
    return {"seed": seed, "starts": n, "command_mps": speed, "need_mps": need,
            "success_rate": rate(ok), "falls": int((~r.alive).sum()),
            "median_speed_mps": float(np.median(fwd[r.alive])) if r.alive.any() else None}


def test_pushes(ecfg, ctrl, seed, sizes=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), per=30,
                walk=0.5, push_at=3.0, after=4.0):
    n = per * len(sizes)
    env = make_env(ecfg, n, seed)
    r = Runner(env, ctrl)
    env.set_command(np.arange(n), walk, 0.0, 0.0, heading=r.start_yaw)
    r.run(push_at)
    before = r.alive.copy()
    rng = np.random.default_rng(seed)
    dv = np.repeat(sizes, per)
    ang = rng.uniform(-np.pi, np.pi, n)
    mag = np.array([env.push_force_for_dv(v) for v in dv])
    env.apply_push(np.arange(n), np.stack([mag * np.cos(ang), mag * np.sin(ang), np.zeros(n)], 1))
    r.run(after)
    rows = []
    for s in sizes:
        m = (dv == s) & before
        rows.append({"push_dv_mps": s, "push_force_N": round(float(env.push_force_for_dv(s)), 1),
                     "bodies": int(m.sum()), "recovered_rate": rate(r.alive[m]) if m.any() else None})
    return {"seed": seed, "walk_mps": walk, "push_s": env.cfg.push_s, "after_s": after, "rows": rows}


def ice_models(env, ids, factor=0.05):
    for i in ids:
        m = copy.deepcopy(env.models[i])
        m.pair_friction[env.body.foot_floor_pairs, :2] *= factor
        env.models[i] = m


def gait_error_commands(env, ids, rng):
    """Goals far outside what was practised: the controller's own gait fails."""
    k = len(ids)
    vx = rng.choice([3.5, -2.0], k)
    yaw = rng.uniform(-3.0, 3.0, k)
    for j, i in enumerate(ids):
        env.set_command([i], vx[j], 0.0, yaw[j])


def falls_experiment(ecfg, ctrl, wm, seed, per=60, inject_at=3.0, after=4.0, conditions=("gait", "push", "slip")):
    """Inject one cause per body (answer key) and record features of falls that follow it."""
    n = per * len(conditions)
    env = make_env(ecfg, n, seed)
    mon = SurpriseMonitor(wm, n, window=100, contact_height=env.cfg.contact_height)
    r = Runner(env, ctrl, mon)
    rng = np.random.default_rng(seed)
    env.set_command(np.arange(n), 0.6, 0.0, 0.0, heading=r.start_yaw)
    sur_walk = []
    r.run(inject_at, lambda rr: sur_walk.append(float(np.median(mon.sur[:, (mon.k - 1) % mon.window][rr.alive]))))
    before = r.alive.copy()
    label = np.repeat(conditions, per)
    for c in conditions:
        ids = np.nonzero(label == c)[0]
        if c == "push":
            dv = rng.uniform(1.5, 2.5, len(ids))
            ang = rng.uniform(-np.pi, np.pi, len(ids))
            for j, i in enumerate(ids):
                f = env.push_force_for_dv(dv[j])
                env.apply_push([i], [f * np.cos(ang[j]), f * np.sin(ang[j]), 0.0])
        elif c == "slip":
            ice_models(env, ids)
            env.set_command(ids, 1.2, 0.0, 0.0, heading=r.start_yaw[ids])
        elif c == "gait":
            gait_error_commands(env, ids, rng)
    pre_falls = set(r.fall_features)
    r.run(after)
    rows = []
    for i, f in r.fall_features.items():
        if i in pre_falls or not before[i]:
            continue
        rows.append({"label": str(label[i]), "U": float(f[0]), "S": float(f[1]),
                     "surprise_before_fall": r.fall_surprise[i], "fall_t": float(r.fall_t[i] - inject_at)})
    return rows, float(np.mean(sur_walk)), {c: int(((label == c) & before).sum()) for c in conditions}


def test_fall_causes(ecfg, ctrl, wm, seed, calib_seed):
    calib_rows, _, _ = falls_experiment(ecfg, ctrl, wm, calib_seed, per=90, conditions=("gait",))
    cal = Calibration.fit(np.array([[r["U"], r["S"]] for r in calib_rows])) if len(calib_rows) >= 5 else Calibration()
    rows, sur_walk, injected = falls_experiment(ecfg, ctrl, wm, seed)
    out = {"seed": seed, "calibration_seed": calib_seed, "calibration": cal.to_dict(),
           "calibration_falls": len(calib_rows), "injected": injected,
           "surprise_walking_median": sur_walk}
    if not rows:
        out["falls_scored"] = 0
        return out
    feats = np.array([[r["U"], r["S"]] for r in rows])
    post = posterior(feats, cal)
    pred = [CAUSES[k] for k in post.argmax(1)]
    labels = [r["label"] for r in rows]
    conf = {t: {p: 0 for p in CAUSES} for t in CAUSES}
    for t, p in zip(labels, pred):
        conf[t][p] += 1
    acc = float(np.mean([t == p for t, p in zip(labels, pred)]))
    per_class = {c: (conf[c][c] / max(1, sum(conf[c].values()))) for c in CAUSES}
    out.update({"falls_scored": len(rows), "accuracy": acc, "balanced_accuracy": float(np.mean(
        [per_class[c] for c in CAUSES if sum(conf[c].values())])), "per_class_recall": per_class,
        "confusion_true_by_pred": conf,
        "falls_by_cause": {c: sum(conf[c].values()) for c in CAUSES},
        "surprise_last_0.2s_before_fall_median": float(np.median([r["surprise_before_fall"] for r in rows])),
        "median_features": {c: np.median(feats[np.array(labels) == c], 0).round(3).tolist()
                            for c in CAUSES if c in labels}})
    return out


def main():
    args, cfg = parse(__doc__, lambda ap: (
        ap.add_argument("--policy", default="runs/locomotion/policy/policy.pt"),
        ap.add_argument("--ckpt", default="runs/locomotion/policy/ckpt_last.pt"),
        ap.add_argument("--name", default="eval"),
        ap.add_argument("--only", default=""),
        ap.add_argument("--seed-offset", type=int, default=0)))
    torch.set_num_threads(2)
    ecfg = EnvConfig.from_dict(cfg.get("env").to_dict() if cfg.get("env") else {})
    s0 = int(cfg.get("eval", {}).get("seeds_start", 50000)) + args.seed_offset
    ctrl = LocomotionController(Path(args.policy))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    wm = EnsembleBodyModel(ck["train_cfg"]["wm_members"])
    wm.load_state_dict(ck["model"])
    wm.eval()
    only = set(args.only.split(",")) if args.only else None
    out = {"policy": args.policy, "policy_iteration": ctrl.meta.get("it"),
           "policy_stage": ctrl.meta.get("stage"), "policy_steps": ctrl.meta.get("total_steps"),
           "commit": ctrl.meta.get("commit"), "seeds_start": s0}
    tests = [("stand", lambda: test_stand(ecfg, ctrl, s0)),
             ("walk_10m", lambda: test_walk(ecfg, ctrl, s0 + 1)),
             ("headings", lambda: test_headings(ecfg, ctrl, s0 + 2)),
             ("run", lambda: test_run(ecfg, ctrl, s0 + 3)),
             ("speeds", lambda: test_speeds(ecfg, ctrl, s0 + 4)),
             ("pushes", lambda: test_pushes(ecfg, ctrl, s0 + 5)),
             ("fall_causes", lambda: test_fall_causes(ecfg, ctrl, wm, s0 + 6, 60000 + args.seed_offset))]
    res = Path("results/locomotion")
    res.mkdir(parents=True, exist_ok=True)
    for name, fn in tests:
        if only and name not in only:
            continue
        t0 = time.time()
        out[name] = fn()
        out[name]["seconds_wall"] = round(time.time() - t0, 1)
        summary = {k: v for k, v in out[name].items() if not isinstance(v, (list, dict))}
        print(name, json.dumps(summary), flush=True)
        (res / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print("wrote", res / f"{args.name}.json")


if __name__ == "__main__":
    main()
