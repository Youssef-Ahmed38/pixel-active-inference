"""Audit round-2 capture and report a failed pre-training gate without a model verdict."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pai.thinker.hier_data import load_stream
from thinker_hier_collect import digest
from thinker_hier_round2 import PREFIX, DOC, rows, save


def percent(result):
    a, b = result['wilson95']
    return f"{result['k']}/{result['n']} ({result['rate']:.1%}; {a:.1%}–{b:.1%})"


def diagnostics(gate):
    from collections import Counter
    result = {}
    for condition in gate['per_condition']:
        selected = [r for r in gate['records'] if r['assigned_condition'] == condition]
        residuals, phases = [], Counter()
        for row in selected:
            with np.load(row['reference'], allow_pickle=False) as data:
                held = data['holding'].astype(bool)
                phases.update(data['step'].tolist())
                if held.any():
                    residuals.append(float(np.median(np.abs(data['residual'][held, 5]))))
        result[condition] = dict(
            median_holding_belief_steps=float(np.median([r['holding_steps'] for r in selected])),
            episodes_with_slip_event=sum(any(e['type']=='slip' for e in r['events']) for r in selected),
            median_episode_held_absolute_vertical_force_residual_N=float(np.median(residuals)),
            phase_steps=dict(phases))
    calibration = json.loads(Path(PREFIX+'calibration.json').read_text(encoding='utf8'))
    save(PREFIX+'diagnostics.json', dict(per_condition=result,
        calibration_force_floors_N={k: v[3:6] for k, v in calibration['step_floors'].items()},
        interpretation='Descriptive scoring after the failed gate; not fitted or used for behaviour/training. Holding is a belief; simulator slip events are audit heuristics.'))
    return result, calibration


def audit():
    plan = json.loads(Path(PREFIX+'plan.json').read_text(encoding='utf8'))
    for path, sha in plan['preserved_round1'].items():
        if digest(path) != sha:
            raise ValueError('Round-1 file changed: ' + path)
    if hashlib.sha256(DOC.read_bytes()[:plan['round1_doc_bytes']]).hexdigest() != plan['round1_doc_sha256']:
        raise ValueError('Round-1 documentation changed')
    counts = {}
    for split in ('calibration', 'development'):
        records = rows(split)
        streams = []
        for mp in Path('results').glob(f'thinker_hier2_{split}_manifest*.json'):
            manifest = json.loads(mp.read_text(encoding='utf8'))
            for path, sha in manifest['source_hashes'].items():
                if digest(path) != sha:
                    raise ValueError('Collection source changed: ' + path)
            if digest(plan['config']['slice']['world_model']) != manifest['world_model_sha256']:
                raise ValueError('Collection world model changed')
            for row in manifest['episodes']:
                if digest(row['path']) != row['sha256']:
                    raise ValueError('Stream hash changed')
                value = load_stream(row['path'])
                n = len(value['action'])
                if value['state'].shape != (n+1, 19) or value['context'].shape != (n, 10):
                    raise ValueError('Bad transition alignment')
                if not all(np.isfinite(x).all() for x in value.values()):
                    raise ValueError('Nonfinite sensory stream')
                with np.load(row['path'], allow_pickle=False) as raw:
                    if len(raw['q']) != n or len(raw['dq']) != n:
                        raise ValueError('Incomplete proprioception')
                reference = next(r for r in records if r['index'] == row['index'])
                with np.load(reference['reference'], allow_pickle=False) as ref:
                    if len(ref['residual']) != n or not np.array_equal(ref['action'], value['action']):
                        raise ValueError('Reference/sensory transition mismatch')
                streams.append(row['index'])
        if len(streams) != len(set(streams)) or len(streams) != len(records):
            raise ValueError('Duplicate or missing capture')
        counts[split] = len(streams)
    # Reject local path strings in every new JSON artifact.
    import re
    for path in Path('results').glob('thinker_hier2_*.json'):
        if re.search(r'[A-Za-z]:[\\/]|/Users/|/home/', path.read_text(encoding='utf8')):
            raise ValueError('Absolute path in new result')
    # Replay the reference from its recorded sensor evidence; never rerun physics.
    from pai.causes.inference import Calibration, StepEvidence, SurpriseMonitor, combined_sigma
    saved = json.loads(Path(PREFIX+'calibration.json').read_text(encoding='utf8'))
    calibration = Calibration(floor=np.array(saved['floor']), threshold=saved['threshold'],
        step_floors={k: np.array(v) for k, v in saved['step_floors'].items()},
        step_thresholds=saved['step_thresholds'])
    gate = json.loads(Path(PREFIX+'gate.json').read_text(encoding='utf8'))
    for row in gate['records']:
        monitor = SurpriseMonitor(.1, calibration=calibration)
        with np.load(row['reference'], allow_pickle=False) as ref:
            for t, (res, sig, step, act, held) in enumerate(zip(ref['residual'], ref['sigma'],
                                                    ref['step'], ref['action'], ref['holding'])):
                ev = StepEvidence(t, res, combined_sigma(sig, calibration, str(step)), act, bool(held), str(step))
                monitor.add(ev, ev)
        verdict = monitor.episode_verdict()
        inferred = verdict.best if verdict else 'none'
        if inferred != row['inferred'] or row['correct'] != (inferred == row['assigned_condition']):
            raise ValueError('Reference replay differs')
    result = dict(round1_files_unchanged=len(plan['preserved_round1']), round1_text_unchanged=True,
                  audited_streams=counts,
                  collection_sources_and_world_model_unchanged=True,
                  reference_source_sha256=digest('pai/causes/inference.py'),
                  reference_replays_identical=len(gate['records']),
                  models_trained=len(list(Path('results').glob('thinker_hier2_*.pt'))),
                  final_streams=len(list(Path('results').glob('thinker_hier2_final_*.npz'))))
    save(PREFIX+'audit.json', result)
    return result


def report():
    gate = json.loads(Path(PREFIX+'gate.json').read_text(encoding='utf8'))
    if gate['passed']:
        raise ValueError('Gate passed; complete the registered training/evaluation before reporting a verdict')
    check = audit()
    if check['models_trained'] or check['final_streams']:
        raise ValueError('Failed gate must stop training and final collection')
    tests = json.loads(Path(PREFIX+'tests.json').read_text(encoding='utf8'))
    if tests['exit_code']:
        raise ValueError('Required tests failed')
    calibration = rows('calibration')
    records = gate['records']
    descriptive, noise = diagnostics(gate)
    lines = ['\n### Round-2 development gate outcome\n',
        '**STOP — DATA GATE FAILED; no round-2 hierarchy GO/NO-GO can be measured.** '
        f"The unchanged Bayesian reference classified {percent(gate['accuracy'])}, below the registered "
        '80% threshold. No neural model was trained; no training or final split was collected. '
        'Round 1 retains its registered NO-GO exactly as written. Round 2 stops before testing '
        'the model comparison and therefore provides no new evidence for or against hierarchy.\n',
        '| Assigned condition | Bayesian correct (Wilson 95%) | Placement (Wilson 95%) | Grasp episodes | Holding-belief episodes | Realized disturbances | Full post-onset 20 |',
        '|---|---|---|---:|---:|---:|---:|']
    for condition, r in gate['per_condition'].items():
        lines.append(f"| {condition} | {percent(r['accuracy'])} | {percent(r['placement'])} | "
            f"{r['grasp_episodes']}/40 | {r['holding_episodes']}/40 | "
            f"{r['realized_disturbance_episodes']}/40 | {r['full_post20_episodes']}/40 |")
    lines += ['', 'Placement overall: '+percent(gate['placement'])+'. '
        f"Clean calibration placement: {sum(r['success'] for r in calibration)}/8. "
        'Grasp is scored from the separate simulator event ledger; holding is the agent’s '
        'subgoal belief and must not be read as a verified physical grasp. Clean rows have '
        'no disturbance onset, so their post-onset count is inapplicable. All episode records, '
        'including absent events and failures, are retained.\n',
        '| True assigned condition \\ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition, r in gate['per_condition'].items():
        lines.append('| '+condition+' | '+' | '.join(str(r['inferred'][k]) for k in
            ('none', 'push', 'heavier_object', 'slippery_object', 'camera_shift', 'unknown'))+' |')
    lines += ['', '| Condition | Median holding-belief steps | Episodes with logged slip | Median episode held absolute vertical-force residual (N) |',
              '|---|---:|---:|---:|']
    for condition, value in descriptive.items():
        lines.append(f"| {condition} | {value['median_holding_belief_steps']:.1f} | "
            f"{value['episodes_with_slip_event']}/40 | "
            f"{value['median_episode_held_absolute_vertical_force_residual_N']:.3f} |")
    lines += ['',
        'The heavier-object force residual is observably larger in this descriptive audit '
        f"({descriptive['heavier_object']['median_episode_held_absolute_vertical_force_residual_N']:.2f} N "
        f"versus {descriptive['none']['median_episode_held_absolute_vertical_force_residual_N']:.2f} N clean), "
        'yet none of the 40 heavier episodes is identified by the Bayesian reference. '
        'The matched clean calibration gives vertical-force noise floors of '
        f"{noise['step_floors']['lifted'][5]:.2f} N in `lifted` and "
        f"{noise['step_floors']['over_target'][5]:.2f} N in `over_target`. "
        'Those large floors can hide a several-newton load from the reference. This is a '
        'plausible explanation for its failures, not an independently tested causal '
        'attribution. The force contrast shows why low reference accuracy alone must not '
        'be equated with absence of all useful sensory information. Logged slip events '
        'are simulator audit heuristics; they do not by themselves identify a friction cause. '
        'The full per-phase diagnostics are retained in `results/thinker_hier2_diagnostics.json`.\n',
        'The normal planner budget and settle tail are restored, but that alone '
        'does not establish that the RGB decoder and resulting motor trajectories provide '
        'identifiable cause evidence. This decoder remains the same given geometry/color '
        'interface used in round 1; the approximately 91% historical Bayesian result uses '
        'a different observation regime and is not an expected score on these pixels. '
        'We do not switch to privileged object state, tune the decoder, or recalibrate after '
        'seeing this failed gate. Repairing perception/exposure requires a separately '
        'registered study before any new hierarchy comparison.\n',
        'The development exposure table is a diagnostic, not a proof that the causes are '
        'information-theoretically unobservable: the reference likelihoods themselves can '
        'also fail under a different perception regime. The pre-registered reference gate '
        'is nevertheless binding; the dataset was not used to tune it.\n',
        'A shared-GPU CUDA out-of-memory error interrupted worker 1 at uncommitted index '
        '87 (layout 95007, heavier object), after 43 saved episodes. Its failed shutdown '
        'was terminated explicitly. Worker 0 finished, then worker 1 resumed its 57 '
        'missing keys in a single process with the same settings and seeds. All completed '
        'episodes were kept. See `results/thinker_hier2_collection_recovery.json`; the '
        'failed log is retained with absolute traceback paths redacted. No failed trial '
        'or model was dropped to improve a score.\n',
        'No probe, clustering, refinement, timescale, attenuation or trained-model figure '
        'is reported for round 2: those measurements would require training after a failed '
        'gate, contrary to the stopping rule. A failed quality gate is not a hierarchy NO-GO.\n',
        f"Audit: {check['round1_files_unchanged']} round-1 files and every original documentation "
        f"byte are unchanged; {sum(check['audited_streams'].values())} sensory streams have "
        'matching capture hashes, finite values and aligned action/proprioception/reference '
        f"lengths. All {check['reference_replays_identical']} reference classifications replay "
        f"identically. Required gates: {sum(r['outcome']=='passed' for r in tests['tests'])} tests passed.\n",
        'Reproduce from the repository root, using the project Python and `PYTHONPATH=.;scripts`, '
        '`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (all outputs remain repository-relative):\n',
        '```powershell',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2.py register',
        '# Run both calibration shards (sequentially or in two modest processes):',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2.py collect --split calibration --workers 2 --worker 0',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2.py collect --split calibration --workers 2 --worker 1',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2_run.py',
        '# Actual recovery after a shared-GPU OOM: survivor first, missing shard only:',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2_resume.py',
        '.venv/Scripts/python.exe scripts/thinker_hier_tests.py --basetemp data/pytest_thinker_hier2_20261028 --output results/thinker_hier2_tests.json',
        '.venv/Scripts/python.exe scripts/thinker_hier_round2_report.py',
        '```', '',
        'Completed captures resume without repeating saved episodes. The pytest temporary '
        'directory must be fresh; choose a new directory under `data/` on a second test run. '
        'The report command is idempotent and never rewrites round-1 text.\n']
    marker = '\n### Round-2 development gate outcome\n'
    old = DOC.read_bytes()
    content = '\n'.join(lines).encode('utf8')
    if marker.encode() in old:
        if old.split(marker.encode(), 1)[1] != content.split(marker.encode(), 1)[1]:
            raise ValueError('Existing round-2 report differs; review before replacing')
    else:
        with DOC.open('ab') as handle:
            handle.write(content)
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    report()
