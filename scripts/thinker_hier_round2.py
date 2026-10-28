"""Round-2 registration, normal-budget sensory collection, and pre-training quality gate.

Round-1 sources and results are read only. This entry point never trains a model.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pai.agents import slice_agent
from pai.causes.inference import Calibration, StepEvidence, SurpriseMonitor, calibrate, combined_sigma
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.thinker.hier_analysis import cause, rate
from pai.thinker.hier_sensors import OwnPixelPerception, sensor_vector
from pai.world.model import load_world_model
from slice_eval import CONDITIONS, disturbance_for
from thinker_hier_collect import PassiveMonitor, digest

PREFIX = 'results/thinker_hier2_'
DOC = Path('docs/THINKER_HIERARCHY.md')
MARKER = '\n## Round 2: registered normal-behaviour replication\n'


def save(path, value):
    import re
    text = json.dumps(value, indent=2, allow_nan=False)
    if re.search(r'[A-Za-z]:[\\/]|/Users/|/home/', text):
        raise ValueError('Absolute path in result')
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf8')
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def config():
    # Only rendering settings differ from slice_eval defaults. Every motor budget,
    # episode cap and settle-tail setting comes unmodified from tabletop.yaml.
    return load_config('configs/tabletop.yaml', ['env.render_images=false', 'env.image_size=160'])


def register():
    path = Path(PREFIX + 'plan.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    prior = json.loads(Path('results/thinker_hier_plan.json').read_text(encoding='utf8'))
    old = DOC.read_bytes()
    if MARKER.strip().encode() in old:
        raise ValueError('Registration section already exists without plan')
    preserved = {str(p).replace('\\', '/'): digest(p) for pattern in
                 ('results/thinker_hier_*', 'scripts/thinker_hier_*.py',
                  'pai/thinker/*.py', 'tests/test_thinker_hier*.py')
                 for p in Path('.').glob(pattern) if p.is_file() and p.name != Path(__file__).name}
    plan = dict(round=2, registered_utc=__import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(),
        prior_plan_sha256=digest('results/thinker_hier_plan.json'),
        preserved_round1=preserved, round1_doc_bytes=len(old),
        round1_doc_sha256=hashlib.sha256(old).hexdigest(),
        models=prior['models'], model=prior['model'], flat_width=prior['flat_width'],
        parameters=prior['parameters'], training_seeds=[96000, 96001, 96002],
        steps=prior['steps'], batch_size=prior['batch_size'], sequence_length=prior['sequence_length'],
        lr=prior['lr'], selection=prior['selection'], go_rule=prior['go_rule'],
        mapping=prior['mapping'], representation=prior['representation'], signatures=prior['signatures'],
        data_seeds=dict(calibration=list(range(95200, 95208)), train=list(range(94000, 94200)),
                        development=list(range(95000, 95040)), final=list(range(93010, 93050))),
        counts_per_condition=dict(train=200, development=40, final=40),
        disturbance_rng_bases=dict(train=960000, development=970000, final=980000),
        config=config().to_dict(), adapt=False,
        perception='Round-1 OwnPixelPerception, RGB only with given geometry/calibration; unchanged',
        gate=dict(split='development', minimum_accuracy=.80, n=200,
                  scoring='Bayesian reference on all 200 assigned conditions; unobserved events count against the assigned condition',
                  calibration='Eight separate clean normal-budget episodes, 95200-95207; same calibrate defaults as slice_eval',
                  failure_action='Stop, report exposure and failure; no model training and no final collection'),
        optional_local_updates=False)
    section = MARKER + '''
This section was registered before round-2 collection or model training. Round 1 and its
NO-GO remain unchanged above. The review identifies an observability confound: its poor
motor/perception performance means that result does not resolve the broader hierarchy thesis.

**Collection:** the `configs/tabletop.yaml` motor defaults used by `scripts/slice_eval.py`:
256 MPPI samples, horizon 10, two iterations, 150 planning steps, action repeat 5,
25 settle steps and `watch_settle=true`. Cause-driven adaptation remains off; a passive
monitor records residuals without inferring causes during behaviour. The RGB-only
`OwnPixelPerception` and its given color/shape geometry remain unchanged from round 1.
No privileged-state replacement is allowed to make the gate pass. Record the complete
monitored settle tail as sensory transitions as well as planning transitions.

**Pre-training gate:** calibrate the unchanged Bayesian reference on eight separate clean
episodes (95200–95207), using the standard calibration defaults. On all 200 development
episodes it must classify at least 160/200 (80%) assigned conditions correctly. Report
Wilson 95% intervals, placement, grasp/holding exposure, realized disturbance events and
post-onset coverage per condition. An absent scheduled event still counts against its
assigned condition. If the gate fails, stop without training or final collection; explain
the failure without changing perception, controller, calibration or threshold to pass it.

**Splits:** training layouts 94000–94199 (200/condition), development 95000–95039
(40/condition), final 93010–93049 (40/condition). The final layouts avoid 93000–93002,
which were round-1 optimizer initialization seeds, as well as all round-1 layout seeds.
Round-2 optimizer seeds are 96000/96001/96002. Disturbance RNG bases are 960000/970000/
980000 by split, plus 1000 × condition index + layout index. Calibration uses clean
episodes only and is separate from all three splits. Do not access final episodes before
models and development readouts are frozen.

**Comparison, conditional on passing the gate:** retain the exact round-1 model classes,
31,974/32,084/31,974 parameters, three paired optimizer seeds, 1,200 AdamW steps,
batch 16, windows 24, learning rate 0.001, and fixed last-step selection. Train only from
sensory streams, without labels or onset windows. Retain train-only PCA/KMeans, development
Hungarian mapping and fixed-C logistic probe. GO requires hierarchical minus flat mean
cluster accuracy ≥5 pp, paired training-seed Student-t 95% lower bound >0, **and** lower
overall prediction MSE. Otherwise report NO-GO. Report probe, NMI, full post-onset-20
MSE/adaptation censoring, same-weight refinement on/off, per-level autocorrelation,
attenuation, and iterations versus surprise/condition, with the round-1 caveats.

All new artifacts use `results/thinker_hier2_*`; round-1 files are hash-preserved.
No optional local weight-learning variant and no post-final tuning are planned.
'''
    # Preserve every original byte, including the pre-existing text encoding.
    with DOC.open('ab') as handle:
        handle.write(section.encode('utf8'))
    save(path, plan)
    return plan


class RecordingPerception(OwnPixelPerception):
    """Capture every sensor call, including settle, without altering its output."""
    def reset(self):
        super().reset()
        self.tokens = []
        self.proprio = []

    def __call__(self, obs, prev_ee=None):
        tokens = super().__call__(obs, prev_ee)
        self.tokens.append(tokens.copy())
        self.proprio.append((obs['q'].copy(), obs['dq'].copy()))
        return tokens


def collect(split, worker, workers, device):
    plan = register()
    if split in ('train', 'final'):
        gate = json.loads(Path(PREFIX + 'gate.json').read_text(encoding='utf8'))
        if not gate['passed']:
            raise ValueError('Data gate failed: training/final collection prohibited')
        if split == 'final' and not Path(PREFIX + 'freeze.json').exists():
            raise ValueError('Freeze models/readouts before final collection')
    cfg = config()
    if cfg.to_dict() != plan['config']:
        raise ValueError('Registered collection settings changed')
    seeds = plan['data_seeds'][split]
    conditions = ['none'] if split == 'calibration' else CONDITIONS
    prefix = PREFIX + split
    mp = Path(prefix + f'_manifest{worker}.json')
    tp = Path(prefix + f'_truth{worker}.json')
    sources = {p: digest(p) for p in ('scripts/thinker_hier_round2.py',
        'pai/thinker/hier_sensors.py', 'pai/agents/slice_agent.py', 'configs/tabletop.yaml')}
    meta = dict(split=split, worker=worker, workers=workers, source_hashes=sources,
                world_model_sha256=digest(cfg.slice.world_model), episodes=[], complete=False)
    truth = dict(split=split, episodes=[])
    if mp.exists():
        old = json.loads(mp.read_text(encoding='utf8'))
        for key in ('source_hashes', 'workers', 'world_model_sha256'):
            if old[key] != meta[key]:
                raise ValueError('Changed collection cannot be resumed: ' + key)
        meta = old
        truth = json.loads(tp.read_text(encoding='utf8'))
    wm = load_world_model(cfg.slice.world_model, device)
    env = TabletopEnv(cfg.env, disturbances=[])
    perceive = RecordingPerception(env)
    original = slice_agent.SurpriseMonitor
    slice_agent.SurpriseMonitor = PassiveMonitor
    try:
        for ci, condition in enumerate(conditions):
            for i, seed in enumerate(seeds):
                index = ci * len(seeds) + i
                if index % workers != worker or any(r['index'] == index for r in meta['episodes']):
                    continue
                rngseed = plan['disturbance_rng_bases'].get(split, 990000) + ci * 1000 + i
                disturbances = disturbance_for(condition, np.random.default_rng(rngseed))
                raw, evidence = [], []
                start = time.time()
                record = slice_agent.run_episode(env, wm, cfg, index, seed, disturbances,
                    device=device, perceive=perceive, adapt=False, raw=raw, evidence=evidence)
                # The standard agent calls perception at initial state, each plan action,
                # and each monitored settle step. Evidence supplies actual motor metadata.
                n = len(raw)
                if len(perceive.tokens) != n + 1 or len(evidence) != n:
                    raise ValueError('Incomplete sensory capture, including settle tail')
                phase = np.array([r[2] for r in evidence])
                holding = np.array([r[1] for r in evidence])
                # Actions are commanded by the standard agent; tail holds still/open.
                action = np.concatenate([record.actions, np.tile([0., 0., 0., 1.],
                                            (n - len(record.actions), 1))]).astype(np.float32)
                path = Path(prefix + f'_{index:04d}.npz')
                np.savez_compressed(path,
                    state=np.stack([sensor_vector(t, f) for t, f in zip(perceive.tokens, perceive.features)]),
                    action=action, context=np.c_[holding, np.eye(9)[np.minimum(phase, 8)]],
                    tokens=np.stack(perceive.tokens),
                    q=np.stack([r[0] for r in perceive.proprio[1:]]),
                    dq=np.stack([r[1] for r in perceive.proprio[1:]]))
                ref = Path(prefix + f'_reference_{index:04d}.npz')
                np.savez_compressed(ref, residual=np.stack([r[0] for r in raw]),
                    sigma=np.stack([r[1] for r in raw]), step=np.array([r[2] for r in raw]),
                    holding=holding, phase=phase, action=action)
                row = dict(index=index, seed=seed, assigned_condition=condition,
                    disturbances=record.true_disturbances, events=record.events, success=record.success,
                    reference=str(ref).replace('\\', '/'), steps=n, holding_steps=int(holding.sum()),
                    settle_steps=sum(r[2] == slice_agent.SETTLE_STEP for r in raw),
                    collection_s=time.time()-start)
                truth['episodes'] = [r for r in truth['episodes'] if r['index'] != index] + [row]
                meta['episodes'].append(dict(index=index, path=str(path).replace('\\', '/'),
                                             sha256=digest(path), steps=n))
                save(tp, truth)
                save(mp, meta)
                print(split, index, condition, seed, n, record.success,
                      'seconds', round(time.time()-start, 1), flush=True)
        meta['complete'] = True
        save(mp, meta)
    finally:
        slice_agent.SurpriseMonitor = original
        env.close()


def rows(split):
    manifests = list(Path('results').glob(f'thinker_hier2_{split}_manifest*.json'))
    if not manifests or not all(json.loads(p.read_text(encoding='utf8'))['complete'] for p in manifests):
        raise ValueError('Incomplete split: ' + split)
    result = []
    for path in Path('results').glob(f'thinker_hier2_{split}_truth*.json'):
        result += json.loads(path.read_text(encoding='utf8'))['episodes']
    if len({r['index'] for r in result}) != len(result):
        raise ValueError('Duplicate episode keys')
    return sorted(result, key=lambda r: r['index'])


def gate():
    plan = register()
    residual, sigma, names = [], [], []
    for row in rows('calibration'):
        with np.load(row['reference'], allow_pickle=False) as data:
            residual.extend(data['residual']); sigma.extend(data['sigma']); names.extend(data['step'])
    c = calibrate(np.stack(residual), np.stack(sigma), names)
    save(PREFIX + 'calibration.json', dict(floor=c.floor.tolist(), threshold=c.threshold,
        step_floors={k: v.tolist() for k, v in c.step_floors.items()}, step_thresholds=c.step_thresholds,
        samples=len(residual), seeds=plan['data_seeds']['calibration']))
    records = []
    for row in rows('development'):
        with np.load(row['reference'], allow_pickle=False) as data:
            monitor = SurpriseMonitor(.1, calibration=c)
            for t, (res, sig, step, act, held) in enumerate(zip(data['residual'], data['sigma'],
                                                 data['step'], data['action'], data['holding'])):
                ev = StepEvidence(t, res, combined_sigma(sig, c, str(step)), act, bool(held), str(step))
                monitor.add(ev, ev)
            report = monitor.episode_verdict()
        inferred = report.best if report else 'none'
        onset = min((max(0, (int(e['t'])-1)//5) for e in row['disturbances']), default=None)
        records.append(dict(**row, realized_cause=cause(row), inferred=inferred,
            correct=inferred == row['assigned_condition'],
            post_onset_steps=max(0, row['steps']-onset) if onset is not None else None,
            grasp=any(e['type'] == 'grasp' for e in row['events'])))
    if len(records) != 200:
        raise ValueError('Gate requires all 200 registered development episodes')
    result = dict(accuracy=rate(sum(r['correct'] for r in records), len(records)),
                  passed=sum(r['correct'] for r in records) >= 160, records=records,
                  per_condition={}, placement=rate(sum(r['success'] for r in records), len(records)))
    for condition in CONDITIONS:
        rs = [r for r in records if r['assigned_condition'] == condition]
        result['per_condition'][condition] = dict(
            accuracy=rate(sum(r['correct'] for r in rs), len(rs)),
            placement=rate(sum(r['success'] for r in rs), len(rs)),
            grasp_episodes=sum(r['grasp'] for r in rs), holding_episodes=sum(r['holding_steps'] > 0 for r in rs),
            realized_disturbance_episodes=sum(bool(r['disturbances']) for r in rs),
            full_post20_episodes=sum(r['post_onset_steps'] is not None and r['post_onset_steps'] >= 20 for r in rs),
            inferred={k: sum(r['inferred'] == k for r in rs) for k in (*CONDITIONS, 'unknown')})
    save(PREFIX + 'gate.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['register', 'collect', 'gate'])
    ap.add_argument('--split', choices=['calibration', 'development', 'train', 'final'])
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'register':
        print(json.dumps(register(), indent=2))
    elif args.stage == 'collect':
        collect(args.split, args.worker, args.workers, args.device)
    else:
        gate()
