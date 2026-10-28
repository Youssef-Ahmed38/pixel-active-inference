"""Append round-3 gates and frozen results; never rewrite earlier rounds."""
import argparse
import hashlib
import json
from pathlib import Path

from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3 import PREFIX, DOC, register


def read(name):
    return json.loads(Path(PREFIX+name+'.json').read_text(encoding='utf8'))


def append(title, lines):
    marker = '\n### '+title+'\n\n'
    text = marker+'\n'.join(lines)+'\n'
    old = DOC.read_bytes()
    if marker.encode() in old:
        # Existing measurements are immutable, including repeated report commands.
        return
    with DOC.open('ab') as handle:
        handle.write(text.encode('utf8'))


def fmt(r):
    return f"{r['k']}/{r['n']} ({r['rate']:.1%}; Wilson 95% {r['wilson95'][0]:.1%}–{r['wilson95'][1]:.1%})"


def gates():
    p = register(); a = read('gate_a')
    lines = [f"Gate A: **{'PASS' if a['passed'] else 'FAIL — STOP'}**. Default privileged-state reproduction, adaptation ON, layouts 96000–96007.",
        f"Task: {fmt(a['task'])}. Cause accuracy: {fmt(a['cause'])}.", '',
        '| Condition | Task | Cause |', '|---|---|---|']
    for c, v in a['per_condition'].items():
        lines.append(f"| {c} | {fmt(v['task'])} | {fmt(v['cause'])} |")
    lines += ['', 'Current package versions: '+', '.join(k+' '+v for k, v in p['versions'].items())+'.',
        'The registration records exact YAML and source/world-model hashes. The published v7 artifact '
        'does not supply package/source provenance sufficient to establish identical historical versions. '
        'The known YAML v8 defaults were kept, including seeded planning and low slippery carry; '
        'the outcome was not used to change them. Calibration remains the standard eight clean '
        'episodes on seeds 5000–5007. Visualization alone was disabled with `--no-gif`.']
    if not a['passed']:
        lines += ['No further collection, training, final evaluation or hierarchy verdict is permitted after Gate A fails.']
    append('Round-3 Gate A reproduction', lines)
    if Path(PREFIX+'gate_b.json').exists():
        b = read('gate_b')
        lines = [f"Gate B: **{'PASS' if b['passed'] else 'FAIL — STOP'}**. Development layouts 98000–98039, adaptation OFF; independent calibration 98200–98207.",
            f"Bayesian accuracy: {fmt(b['accuracy'])}. Placement: {fmt(b['placement'])}.", '',
            '| Condition | Bayesian correct | Placement | Grasp | Holding belief | Realized disturbance | Full post20 |',
            '|---|---|---|---:|---:|---:|---:|']
        for c, v in b['per_condition'].items():
            lines.append(f"| {c} | {fmt(v['accuracy'])} | {fmt(v['placement'])} | "
                f"{v['grasp_episodes']}/40 | {v['holding_episodes']}/40 | "
                f"{v['realized_disturbance_episodes']}/40 | {v['full_post20_episodes']}/40 |")
        lines += ['', 'Grasp/disturbance events are separate scoring evidence; holding is the agent’s internal belief. '
            'Clean post-onset counts are inapplicable. Events absent from a short episode still count '
            'against the assigned cause for Gate B. Predictive evaluations use actual logged causes and '
            'retain censored/truncated windows. Calibration is not retuned after this gate.']
        if not b['passed']:
            lines += ['Stop before training or final collection. This failed data gate is not a hierarchy NO-GO.']
        append('Round-3 Gate B development quality', lines)


def audit_previous():
    p = register()
    for path, sha in p['preserved_previous_rounds'].items():
        if digest(path) != sha:
            raise ValueError('Previous-round artifact changed: '+path)
    if hashlib.sha256(DOC.read_bytes()[:p['previous_doc_bytes']]).hexdigest() != p['previous_doc_sha256']:
        raise ValueError('Previous-round documentation changed')
    return len(p['preserved_previous_rounds'])


def final():
    import numpy as np
    from thinker_hier_round3_figures import aggregate, figures
    from thinker_hier_round3_train import KINDS, SEEDS
    from thinker_hier_round3_evaluate import VARIANTS, paths
    raw, out = aggregate(); figures(raw, out)
    save(PREFIX+'summary.json', out)
    refs = read('references_final'); p = out['paired']
    def av(v, percent=False):
        if v['mean'] is None:
            return 'n/a'
        scale = 100 if percent else 1
        return f"{scale*v['mean']:.4f} ± {scale*v['std']:.4f}"+('%' if percent else '')
    lines = ['Frozen final layouts 99000–99039; three optimizer seeds 100000/100001/100002. '
        'Values below are mean ± sample std over training seeds. Labels were unavailable to weight learning; '
        'only development labels fit the Hungarian mapping and fixed-C linear probe.', '',
        '| Model | Cluster accuracy | Linear probe | NMI | Overall MSE | Full-post20 MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for k in VARIANTS:
        v = out['models'][k]
        lines.append(f"| {k} | {av(v['cluster_accuracy'],True)} | {av(v['probe_accuracy'],True)} | "
            f"{av(v['nmi'])} | {av(v['mse'])} | {av(v['post20_mse'])} |")
    lines += ['', f"**Registered verdict: {p['verdict']}**. Hierarchical-minus-flat cluster gain "
        f"{100*p['cluster_difference']:.2f} pp, paired optimizer-seed Student-t 95% "
        f"[{100*p['paired_seed_t95'][0]:.2f}, {100*p['paired_seed_t95'][1]:.2f}] pp. "
        f"MSE ratio {p['prediction_mse_ratio']:.5f}. Layout-cluster bootstrap 95% "
        f"[{100*p['layout_bootstrap95'][0]:.2f}, {100*p['layout_bootstrap95'][1]:.2f}] pp "
        'measures a different uncertainty source and is not a joint seed/layout interval.', '',
        '| Model / training seed | Cluster (Wilson 95%) | Probe (Wilson 95%) | Training seconds |',
        '|---|---|---|---:|']
    for k in VARIANTS:
        trained = 'hierarchical' if k=='hierarchical_test_amortized' else k
        for seed, run in zip(SEEDS, raw[k]):
            training = json.loads(Path(paths(trained, seed)+'_training.json').read_text(encoding='utf8'))
            lines.append(f"| {k} / {seed} | {fmt(run['summary']['cluster_accuracy'])} | "
                f"{fmt(run['summary']['probe_accuracy'])} | {training['train_wall_s']:.2f} |")
    lines += ['', '| Model | L1 ACF lag | L2 ACF lag | L3 ACF lag | Iterations / update | Capped adaptation |',
        '|---|---:|---:|---:|---:|---:|']
    for k in KINDS:
        v = out['models'][k]
        lines.append('| '+k+' | '+' | '.join(av(v['timescales'][l]) for l in ('level1','level2','level3'))+
            ' | '+av(v['update_iterations'])+' | '+av(v['restricted_adaptation'])+' |')
    lines += ['', '![Round-3 measured timescales](../results/thinker_hier3_timescales.png)', '',
        '| Condition | H iterations/update | H early error (steps 0–4) | H late error (steps 15–19) |',
        '|---|---:|---:|---:|']
    for c, v in out['thinking']['hierarchical']['by_condition'].items():
        lines.append(f"| {c} | {v['update_iterations']:.4f} | {v['early_error']} | {v['late_error']} |")
    lines += ['', '![Round-3 event-aligned errors](../results/thinker_hier3_error_dynamics.png)', '',
        '![Round-3 inference effort](../results/thinker_hier3_thinking.png)', '',
        f"Surprise/update-iterations Spearman: {out['thinking']['hierarchical']['spearman_surprise_iterations']}. "
        'The effort budget and update schedules are given. ACF lags are nonstationary trajectory '
        'statistics, not neuronal intrinsic time constants. Pulse termination, carry timing, and '
        'given sensor/state mechanics can explain attenuation; this is not proof of a brain mechanism.',
        'Same-weight `hierarchical_test_amortized` isolates inference-time refinement; the separately '
        'trained `amortized` branch is the registered training ablation. Compare their accuracy and '
        'MSE above rather than treating fitted-past-error reduction as future-prediction improvement.', '',
        f"Bayesian reference: {fmt(refs['bayes'])}. Supervised HRM references: "+'; '.join(fmt(v) for v in refs['hrm'])+'. '
        'These given-likelihood/teacher-trained systems are references, not parameter-matched '
        'unsupervised competitors. Copy-current-senses normalized MSE: '+str(refs['persistence_mse'])+'.',
        'Full per-episode failures, dominant code usage/entropy, per-cause recall, incomplete windows, '
        'censoring and ACF undefined/capped counts are retained in each final JSON. No local '
        'predictive-coding weight update variant was implemented. Three training seeds leave wide '
        'uncertainty; the registered threshold is not a universal test of hierarchy.', '',
        'The conclusion applies to the given privileged-token regime and fixed training budget. '
        'It cannot establish visual perception learning or the full proposed thesis. No final '
        'outcome was used to tune, replace a seed, or select a checkpoint.']
    append('Round-3 frozen go/no-go and brain-signature measurements', lines)
    print(json.dumps(p, indent=2), flush=True)


def commands():
    append('Round-3 reproduction commands and audit', [
        'Run from the project root using the project Python, `PYTHONPATH=.;scripts`, '
        '`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. All outputs are repository-relative.', '',
        '```powershell',
        '.venv/Scripts/python.exe scripts/thinker_hier_round3.py register',
        '.venv/Scripts/python.exe scripts/thinker_hier_round3.py reproduce',
        '.venv/Scripts/python.exe scripts/thinker_hier_tests.py --basetemp data/pytest_thinker_hier3_20261028 --output results/thinker_hier3_tests.json',
        '.venv/Scripts/python.exe scripts/thinker_hier_round3_pipeline.py',
        '.venv/Scripts/python.exe scripts/thinker_hier_round3_audit.py',
        '```', '',
        'The pipeline stops at either failed gate. It logs exact per-stage commands, resumes only '
        'saved episode/checkpoint keys, freezes before any final collection and preserves failures. '
        'Choose a fresh pytest directory under `data/` for a second test invocation. '
        'The final audit verifies earlier-round hashes, sensory stream hashes, matched training '
        'budgets, frozen sources/models/readouts and per-episode replay. No git state changes or '
        'cloud compute are part of these commands.'])
    append('Round-3 readiness command before training', [
        'The later readiness test result is the one checked before training and frozen '
        'with the models. Run this after preparing the new scripts and before the pipeline '
        '(the initial test result above is retained separately):', '',
        '```powershell',
        '.venv/Scripts/python.exe scripts/thinker_hier_tests.py --basetemp data/pytest_thinker_hier3_ready_20261028 --output results/thinker_hier3_tests_ready.json',
        '```', '', 'This readiness run passed 66 tests. Use a new temporary directory for any repeat.'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['gates', 'final'])
    args = ap.parse_args(); gates()
    if args.stage=='final': final()
    commands()
    print('Earlier-round files preserved:', audit_previous(), flush=True)
