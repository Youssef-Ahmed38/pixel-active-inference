"""Round-3 pre-registration and literal default slice reproduction (Gate A)."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from pai.config import load_config
from pai.thinker.hier_analysis import rate
from thinker_hier_collect import digest
from thinker_hier_round2 import save

PREFIX = 'results/thinker_hier3_'
DOC = Path('docs/THINKER_HIERARCHY.md')


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'mujoco', 'scipy', 'scikit-learn')}


def register():
    path = Path(PREFIX+'plan.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    original = DOC.read_bytes()
    if b'\n## Round 3:' in original:
        raise ValueError('Round-3 section exists without registration')
    prior = json.loads(Path('results/thinker_hier_plan.json').read_text(encoding='utf8'))
    cfg = load_config('configs/tabletop.yaml')
    preserved = {p.as_posix(): digest(p) for pattern in
        ('results/thinker_hier_*', 'results/thinker_hier2_*',
         'scripts/thinker_hier_*.py', 'pai/thinker/*.py', 'tests/test_thinker_hier*.py')
        for p in Path('.').glob(pattern) if p.is_file() and p.name != Path(__file__).name}
    command = ['scripts/slice_eval.py', '--config', 'configs/tabletop.yaml', '--episode-seed', '96000',
               '--calibration-episodes', '8', '--no-gif', '--tag', 'thinker_hier3_reproduction', 'slice.episodes=8']
    plan = dict(round=3, registered_utc=__import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat(), preserved_previous_rounds=preserved,
        previous_doc_bytes=len(original), previous_doc_sha256=hashlib.sha256(original).hexdigest(),
        reproduction_command=command, reproduction_config=cfg.to_dict(), versions=versions(),
        registered_sources={p: digest(p) for p in ('scripts/slice_eval.py', 'pai/agents/slice_agent.py',
            'pai/causes/inference.py', 'pai/perception/camera_frame.py', 'configs/tabletop.yaml')},
        world_model_sha256=digest(cfg.slice.world_model),
        gate_a=dict(n=40, task_minimum=.8, cause_minimum=.8,
                    stop='Stop before collection/training if either count is below 32/40'),
        gate_b=dict(n=200, bayesian_minimum=.8,
                    stop='Stop before training/final collection if Bayesian count is below 160/200'),
        data_seeds=dict(train=list(range(97000, 97200)), development=list(range(98000, 98040)),
                        final=list(range(99000, 99040)), calibration=list(range(98200, 98208))),
        disturbance_rng_bases=dict(train=1000000, development=1010000, final=1020000),
        training_seeds=[100000, 100001, 100002],
        models=prior['models'], model=prior['model'], parameters=prior['parameters'], flat_width=prior['flat_width'],
        steps=prior['steps'], batch_size=prior['batch_size'], sequence_length=prior['sequence_length'],
        lr=prior['lr'], selection=prior['selection'], go_rule=prior['go_rule'],
        mapping=prior['mapping'], representation=prior['representation'], signatures=prior['signatures'],
        sensory_channels='gripper xyz, wrist force xyz, perceived target xyz, mean other-object xyz, gripper opening, target velocity xyz, mean other-object velocity xyz (19); actions 4; holding+phase 10',
        given='Privileged object entity tokens via the default CameraFramePerception; learned arm world model and standard MPPI. No RGB decoder.',
        collection_adaptation=False, optional_local_updates=False)
    section = '''
## Round 3: registered privileged-token experiment

Registered before reproduction or new collection. All earlier results and text remain
unchanged. Round 3 addresses the reviewed perception mismatch explicitly: **object state
is given (privileged)** through the arm's default `CameraFramePerception`. The thinker
would learn from the arm's own entity-token/action/force stream, **not from pixels**.
No round-1/2 RGB geometric decoder is used. This narrower experiment cannot establish
learned visual perception or the paper's full perception-learning thesis.

**Gate A, reproduce first:** run the unchanged `scripts/slice_eval.py` with
`--config configs/tabletop.yaml --episode-seed 96000 --calibration-episodes 8 --no-gif
--tag thinker_hier3_reproduction slice.episodes=8`. There are eight episodes per each
of five conditions (40 total); cause adaptation is ON, as in the published evaluation.
Calibration uses the script's unchanged seeds 5000–5007 and disturbance RNG seed 0.
`--no-gif` suppresses visualization/rendering, without changing the privileged tokens.
No planner, physics, episode-length, perception, or calibration override is allowed.
Pass only if task success **and** cause accuracy each reach 32/40 (80%). Otherwise stop,
document configuration/version differences and do not collect or train anything else.

The checked-in `results/slice_v7/slice_eval.md` actually lists task success
100/100/90/75/90% by condition (91% overall), and cause accuracy 100/90/90/60/100%
(88% overall); its slippery task row is 75%, rather than the review's 90–100% range.
The current YAML includes later v8 `seed_planner=true`, `slippery_carry=low`, and
`watch_settle=true` defaults. We will use these current defaults without reverting or
tuning them. Historical package/source provenance may be unavailable; current package,
source, config and world-model hashes are registered in `thinker_hier3_plan.json`.

**Collection, only after Gate A:** exactly the same default token perception, world model,
MPPI and episode settings, with adaptation OFF (`--no-adapt` behaviour) and optional
sensory recording. Cause labels/events never enter the predictor or motor commands.
Save labels/events separately for scoring. Capture planning and monitored settle
transitions; any recording wrapper must return unchanged default perception outputs.
The 19 predicted channels are gripper xyz, wrist force xyz, target xyz, mean other-object
xyz, gripper opening, target velocity xyz and mean other-object velocity xyz. Actions
have four channels and holding/phase context ten, retaining exactly the round-1 model
dimensions and parameter budgets. The last six channels are now token velocities,
replacing the round-1 RGB summary; no pixel information is supplied or claimed.

**Gate B:** calibrate the unchanged Bayesian reference on separate clean layouts
98200–98207, using standard calibration settings, then evaluate all 200 development
episodes (40/condition, layouts 98000–98039). It must classify at least 160/200 assigned
conditions (80%). Report placement and grasp/holding/disturbance/post-onset exposure per
condition, with Wilson 95% intervals. Absent scheduled events count against the assigned
condition. If Gate B fails, stop before training and final collection, without tuning.

**Splits and frozen comparison, conditional on both gates passing:** collect 200 training
episodes/condition on layouts 97000–97199; development 98000–98039; final 99000–99039.
Final data remain unopened until models/readouts are frozen. Disturbance RNG bases are
1000000/1010000/1020000 by split, plus 1000 × condition index + layout index. Optimizer
seeds are 100000/100001/100002. Use the identical three round-1 model classes and
31,974/32,084/31,974 parameter counts, 1,200 AdamW steps, batch 16, windows 24,
learning rate .001 and fixed last-step selection. Train only on training sensory streams.
Retain train-only PCA/KMeans and development-only Hungarian mapping/logistic probe.
GO requires hierarchical-minus-flat mean cluster accuracy ≥5 pp, paired optimizer-seed
Student-t 95% lower bound >0, and lower overall prediction MSE. Otherwise NO-GO.

Report accuracy/NMI/probe with seed uncertainty, Wilson intervals where counts,
full-post-onset-20 prediction/adaptation with censoring, same-weight refinement on/off,
state autocorrelations, attenuation, iterations versus own surprise/condition, and wall
time, retaining all round-1 caveats (scheduled timescales, imposed iteration budget,
nonstationary trajectories, physical exposure timing and pulse duration). No optional
local weight-learning variant or post-final tuning. All new results use
`results/thinker_hier3_*`, repository-relative paths, and laptop compute only.
'''
    with DOC.open('ab') as handle:
        handle.write(section.encode('utf8'))
    save(path, plan)
    return plan


def reproduce():
    plan = register()
    output = Path(PREFIX+'reproduction_eval.json')
    if output.exists():
        raise FileExistsError('Reproduction already completed; score existing records, do not rerun')
    for path, sha in plan['registered_sources'].items():
        if digest(path) != sha:
            raise ValueError('Registered source changed: '+path)
    with Path(PREFIX+'reproduction.log').open('a', encoding='utf8') as log:
        subprocess.run([sys.executable, *plan['reproduction_command']],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    score_a()


def score_a():
    plan = register()
    value = json.loads(Path(PREFIX+'reproduction_eval.json').read_text(encoding='utf8'))
    rows = value['episodes']
    if len(rows) != 40:
        raise ValueError('Gate A requires all 40 episodes')
    success = sum(r['success'] for r in rows)
    correct = sum(r['condition'] == r['inferred'] for r in rows)
    result = dict(passed=success >= 32 and correct >= 32,
        task=rate(success, 40), cause=rate(correct, 40), per_condition={},
        command=['python', *plan['reproduction_command']], versions=versions())
    for condition in ('none', 'push', 'heavier_object', 'slippery_object', 'camera_shift'):
        rs = [r for r in rows if r['condition'] == condition]
        result['per_condition'][condition] = dict(task=rate(sum(r['success'] for r in rs), len(rs)),
            cause=rate(sum(r['inferred'] == condition for r in rs), len(rs)))
    save(PREFIX+'gate_a.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['register', 'reproduce', 'score-a'])
    args = ap.parse_args()
    if args.stage == 'register':
        register(); print('Round 3 registered before collection', flush=True)
    elif args.stage == 'reproduce':
        reproduce()
    else:
        score_a()
