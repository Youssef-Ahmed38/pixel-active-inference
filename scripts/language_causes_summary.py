"""Audit and summarize frozen causal-language evaluation without rerunning physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from pai.language.causal import ClauseLanguage, CausalExplainer, EpisodeEvidence


def summarize(report):
    rows = report['episodes']
    keys = [(r['seed'],r['condition'],r['mode']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate physical trials')
    conditions = ('none','push','heavier_object','slippery_object','camera_shift')
    expected = {(seed,c,m) for seed in report['seeds'] for c in conditions
                for m in ('baseline','english','arabic')}
    if not report['complete'] or set(keys) != expected:
        raise ValueError('Incomplete evaluation')
    replay_count = 0
    for r in rows:
        evidence = EpisodeEvidence(**r['belief'])
        for lang, templates in report['learned_templates'].items():
            ex = CausalExplainer(ClauseLanguage(templates),report['threshold'])
            saved = r['outputs'][lang]
            if ex.explain(evidence) != saved['explanation']:
                raise ValueError('Explanation cannot be reproduced from agent evidence')
            truth_causes = {dict(payload='heavier_object',friction='slippery_object').get(
                e['disturbance'],e['disturbance']) for e in r['truth']['disturbances']} or {'none'}
            correct = saved['explanation']['cause'] in truth_causes
            if (saved['score']['truth_causes'] != sorted(truth_causes)
                or saved['score']['cause_correct'] != correct
                or saved['score']['goal_correct'] != (evidence.goal_observed == r['truth']['success'])):
                raise ValueError('Saved scoring disagrees with raw truth')
            checks = {}
            estimates = saved['explanation']['estimates']
            for event in r['truth']['disturbances']:
                prm = event['params']
                if 'extra_mass_kg' in estimates and 'mass' in prm:
                    checks['extra_mass_kg'] = (abs(estimates['extra_mass_kg']-prm['mass']),.1)
                if 'camera_offset_mm' in estimates and 'offset' in prm:
                    error = math.sqrt(sum((a-1000*b)**2 for a,b in zip(estimates['camera_offset_mm'],prm['offset'])))
                    checks['camera_offset_mm'] = (error,10.)
                if 'onset' in estimates:
                    dt = report['config']['env']['control_dt']
                    checks['onset'] = (abs(estimates['onset']*dt*report['config']['slice']['action_repeat']-event['t']*dt),.3)
            if set(checks) != set(saved['score']['parameters']):
                raise ValueError('Missing or extraneous parameter score')
            for key,(error,tolerance) in checks.items():
                check = saved['score']['parameters'][key]
                if (not math.isclose(check['error'],error,abs_tol=1e-9)
                    or check['tolerance'] != tolerance or check['passed'] != (correct and error <= tolerance)):
                    raise ValueError('Parameter score disagrees with raw truth')
            for key, question in saved['questions'].items():
                if ex.answer(question,evidence) != saved['answers'][key]:
                    raise ValueError('Answer cannot be reproduced from agent evidence')
            replay_count += 1
    result = dict(physical_trials=len(rows), independent_layouts=len(report['seeds']),
                  matched_layout_condition_pairs=len(report['seeds'])*5,
                  replayed_bilingual_outputs=replay_count, languages={},conditions={},
                  qa_scope='Held-out episodes and episode-number slots in taught question grammar; no novel paraphrase claim')
    for condition in conditions:
        result['conditions'][condition] = {}
        for mode in ('baseline','english','arabic'):
            group = [r for r in rows if r['condition']==condition and r['mode']==mode]
            result['conditions'][condition][mode] = dict(n=len(group),success=sum(r['truth']['success'] for r in group),
                goal_observation_correct=sum(r['outputs']['english']['score']['goal_correct'] for r in group))
    for lang in ('english','arabic'):
        # Each language's own executed trials, not repeated translations of all trials.
        group = [r for r in rows if r['mode']==lang]
        out = [r['outputs'][lang] for r in group]
        accepted = [o for o in out if not o['explanation']['abstained']]
        parameters = {}
        for o in out:
            for key, check in o['score']['parameters'].items():
                p = parameters.setdefault(key,dict(n=0,passed=0,errors=[]))
                p['n'] += 1
                p['passed'] += check['passed']
                p['errors'].append(check['error'])
        for p in parameters.values():
            p['mean_error'] = sum(p['errors'])/p['n']
        qa = {key:sum(o['qa'][key] for o in out) for key in out[0]['qa']}
        condition_scores = {}
        for condition in conditions:
            subset = [r['outputs'][lang] for r in group if r['condition']==condition]
            condition_scores[condition] = dict(n=len(subset),cause_correct=sum(o['score']['cause_correct'] for o in subset),
                accepted=sum(not o['explanation']['abstained'] for o in subset))
        baseline = {(r['seed'],r['condition']):r for r in rows if r['mode']=='baseline'}
        result['languages'][lang] = dict(n=len(out),accepted=len(accepted),
             cause_correct=sum(o['score']['cause_correct'] for o in out),
             correct_accepted=sum(o['score']['cause_correct'] for o in accepted),
             abstentions=len(out)-len(accepted),
             clean_abstentions=sum(r['condition']=='none' and r['outputs'][lang]['explanation']['abstained'] for r in group),
             disturbed_abstentions=sum(r['condition']!='none' and r['outputs'][lang]['explanation']['abstained'] for r in group),
             goal_correct=sum(o['score']['goal_correct'] for o in out),parameters=parameters,qa=qa,
             supported_counterfactual_answers=sum(not o['answers']['cf:open_early']['abstained'] for o in out),
             why_cause_correct=sum(o['answers']['why'].get('cause') in o['score']['truth_causes'] for o in out),
             by_condition=condition_scores,
             baseline_success=sum(baseline[(r['seed'],r['condition'])]['truth']['success'] for r in group),
             instruction_success=sum(r['truth']['success'] for r in group),
             paired_success_disagreements=sum(r['truth']['success']!=baseline[(r['seed'],r['condition'])]['truth']['success'] for r in group))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args = ap.parse_args()
    report = json.loads(args.input.read_text(encoding='utf-8'))
    result = summarize(report)
    result['source'] = str(args.input)
    result['source_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['command'] = [sys.executable,*sys.argv]
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
