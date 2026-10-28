"""Replay V4 and frozen V3 on identical evidence; calibration, paired tests, inventory."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from pai.language.causal import EpisodeEvidence
from pai.language.causal_v4 import load_v4,CalibratedSilence
from language_causes_v2_eval import load_explainers,score_v2
from language_causes_v2_summary import rate
from language_causes_v3_paired import sign_test
from language_causes_v4_train import save,digest
from language_causes_v4_score import score_v4


def reliability(p,y):
    rows=[]
    for low,high in zip((0.,.2,.4,.6,.8),(.2,.4,.6,.8,1.)):
        ids=[i for i,v in enumerate(p) if low<=v and (v<high or high==1.)]
        rows.append(dict(bin=[low,high],n=len(ids),mean_confidence=float(np.mean([p[i] for i in ids])) if ids else None,
                         observed_accuracy=rate(sum(y[i] for i in ids),len(ids))))
    return dict(n=len(p),brier=float(np.mean((np.asarray(p)-np.asarray(y))**2)) if p else None,bins=rows)


def none_probability(ex,evidence):
    # Only beliefs, including the learned calibration model, enter forecasts.
    if hasattr(ex,'silence') and isinstance(ex.silence,CalibratedSilence):
        if not evidence.posterior:
            return ex.silence.posterior('silent')['none']
        if max(evidence.posterior,key=evidence.posterior.get)=='none':
            return ex.silence.posterior('explicit_none')['none']
    if not evidence.posterior and ex.silence is not None:
        return ex.silence.p_none_given_silent()
    return evidence.posterior.get('none',0.)


def arabic_inventory(checkpoint):
    m=checkpoint['models']['arabic'];v=checkpoint['v4']['languages']['arabic']
    from pai.language.causal_v4 import LearnedComposition
    composer=LearnedComposition(**v['composition'])
    entries=[]
    for phrase,sentence in v['teacher_pairs']:
        original=next((old for old,new in v['phrases'].items() if phrase==new),None)
        entries.append(dict(key='hypothesis:'+phrase,text=composer.say(phrase),
                            note='learned first-person verb frame; accusative spelling corrected' if phrase.startswith('لم') else
                                 'given gender/number agreement; learned noun/pronoun complement frame',
                            runtime=phrase in [*v['phrases'].values(),v['hedge']]))
    for key,template in m['templates'].items():
        if key=='hypothesis' or key.startswith('cause:'):
            continue
        entries.append(dict(key=key,text=v['replacements'].get(template,template),runtime=True,
                            note='corrected teacher wording/count construction' if template in v['replacements'] else 'unchanged grammatical sentence'))
    for item in m['lexicon']['numeric']:
        if not any(r['text']==item['template'] for r in entries):
            entries.append(dict(key='numeric',text=item['template'],runtime=True,note='unchanged; {value} is an agent estimate'))
    entries.append(dict(key='probability',text=v['probability_frame'],runtime=True,note='new taught probability sentence; {value} is calibrated probability'))
    return entries


def analyze(data,old,new):
    if not data['complete']:
        raise ValueError('Incomplete final')
    expected={(s,c,'baseline') for s in range(69020,69040) for c in
              ('none','push','heavier_object','slippery_object','camera_shift')}
    keys=[(r['seed'],r['condition'],r['mode']) for r in data['episodes']]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Final keys incomplete/duplicated')
    models={'v3':load_explainers(old),'v4':load_v4(new)}
    out=dict(episodes=[],languages={},arabic_inventory=arabic_inventory(new),physical_trials=len(keys),
             calibration_design=new['v4']['diagnostics'])
    for r in data['episodes']:
        ev=EpisodeEvidence(**r['belief'])
        scored={}
        for version,langs in models.items():
            scored[version]={}
            for lang,ex in langs.items():
                explanation=ex.explain(ev)
                score=(score_v4 if version=='v4' else score_v2)(explanation,r['truth'],lang)
                if version=='v4':
                    if explanation!=r['outputs'][lang]['explanation'] or score!=r['outputs'][lang]['score']:
                        raise ValueError('Frozen episode output mismatch')
                    for q in r['outputs'][lang]['qa']:
                        if ex.answer(q['text'],ev)!=q['answer']:
                            raise ValueError('Question replay mismatch')
                scored[version][lang]=dict(explanation=explanation,score=score,none_probability=none_probability(ex,ev))
        out['episodes'].append(dict(seed=r['seed'],condition=r['condition'],belief=r['belief'],truth=r['truth'],outputs=scored))
    for lang in ('english','arabic'):
        result={}
        for version in models:
            os=[r['outputs'][version][lang] for r in out['episodes']]
            issued=[o for o in os if not o['explanation']['abstained']]
            confident=[o for o in issued if o['score']['reported_cause']=='none' and
                       (o['explanation'].get('claim_strength')=='confident' if version=='v4' else o['explanation']['confidence']>=.95)]
            silent=[i for i,r in enumerate(out['episodes']) if not r['belief']['posterior']]
            p=[o['none_probability'] for o in os]
            clean=[not bool(r['truth']['disturbances']) for r in out['episodes']]
            top=[o for o in issued if o['explanation']['confidence'] is not None]
            result[version]=dict(cause_correct=rate(sum(o['score']['cause_correct'] for o in os),len(os)),
                coverage=rate(len(issued),len(os)),precision=rate(sum(o['score']['cause_correct'] for o in issued),len(issued)),
                confident_none_precision=rate(sum(o['score']['cause_correct'] for o in confident),len(confident)),
                hedged_none=sum(o['explanation'].get('claim_strength')=='hedged' for o in os),
                binary_none_calibration=reliability(p,clean),
                silent_none_calibration=reliability([p[i] for i in silent],[clean[i] for i in silent]),
                top_label_calibration=reliability([o['explanation']['confidence'] for o in top],
                                                 [o['score']['cause_correct'] for o in top]),
                by_condition={c:rate(sum(o['score']['cause_correct'] for o,r in zip(os,out['episodes']) if r['condition']==c),
                                      sum(r['condition']==c for r in out['episodes'])) for c in new['v4']['silence']['counts']},
                success=rate(sum(r['truth']['success'] for r in out['episodes']),len(os)))
        a=[r['outputs']['v3'][lang]['score']['cause_correct'] for r in out['episodes']]
        b=[r['outputs']['v4'][lang]['score']['cause_correct'] for r in out['episodes']]
        wins=sum(y and not x for x,y in zip(a,b));losses=sum(x and not y for x,y in zip(a,b))
        result['paired']=dict(only_v4_correct=wins,only_v3_correct=losses,sign_test_p=sign_test(wins,losses))
        out['languages'][lang]=result
    return out



def append_documentation(summary,checkpoint):
    path=Path('docs/LANGUAGE_STAGE6.md')
    original=path.read_text(encoding='utf8')
    if '\n## V4:' in original:
        raise ValueError('V4 section already present; refusing duplicate append')
    def fmt(r):
        if r['n']==0:return '0/0 (not estimable)'
        lo,hi=r['wilson95']
        return f"{r['k']}/{r['n']} [{100*lo:.1f}–{100*hi:.1f}%]"
    lines=['','## V4: learned Arabic composition and calibrated absence claims','',
    'V4 retains the frozen v3 anonymous-coordinate cause and quantity associations, question classifier, arm controller, and surprise monitor. It adds independently taught English/Arabic surface composition and new calibration of monitor silence and explicit `none` decisions. Event logs are supplied only to the teacher and scorer. V4 scoring decodes the actually rendered `surface_phrase`, including the hedge, rather than trusting a latent cause key or an old phrase spelling. Runtime explanations take `EpisodeEvidence` and frozen learned parameters, never truth or disturbance parameters. Original v1–v3 artifacts and documentation are unchanged.','',
    '### Design: learned versus given','',
    'Given: the teacher’s grammatical complete sentences and corrected phrases, semantic evidence schema, balanced five-condition deployment prior, independent Jeffreys Beta(0.5,0.5) likelihood priors, minimum claim probability 0.5 and confident threshold 0.95. Arabic morphology, vocabulary corrections, and the count-label construction are teacher supervision, not independently discovered grammar. The teacher supplies both noun complements and first/third-person verb complements. The learner removes the paired phrase from each sentence, induces distinct frames, and learns each phrase’s frame index without receiving a clause-type label, cause name, or hand-written complementizer-selection rule. Untaught compositions are rejected rather than guessed. This is controlled phrase/frame learning; it does not generalize inflection to arbitrary Arabic. The existing cause-coordinate links remain those learned cross-situationally in v3, independently per language.','',
    'Learned anew: phrase-to-frame choices (including `أنني` for both first-person `none` and hedge clauses, `أنّ` before noun complements, and teacher examples with `أنه`/`أنها`), a probability sentence template induced from two numeric teacher examples, and per-condition frequencies of silence and explicit `none`. The surface phrase corrections and hedge wording are given by the teacher. Numeric probability slots come from the learned calibration model. English disturbed reports remain byte-identical to v3; English absence claims gain matched hedging and a numeric probability. Arabic count reporting uses a grammatical count label for every cardinal rather than an incorrectly invariant `مرة` after arbitrary numerals.','',
    'For condition c and event e, the learned posterior-predictive likelihood is `(count(e,c)+0.5)/(n(c)+1)`. Bayes’ rule normalizes these five likelihoods with an explicitly balanced prior. Pooling four unseen disturbance types into one near-zero pooled likelihood is avoided. Silence and an explicit monitor `none` are calibrated separately. Credible intervals propagate Beta likelihood uncertainty (20,000 fixed diagnostic draws, seed 70420); reported probabilities are ratios of posterior-predictive likelihoods, not lower confidence bounds. Claims below 0.5 abstain, 0.5–0.95 hedge, and at least 0.95 use the confident phrase. Neither thresholds nor priors were selected using final episodes. The probability depends on the balanced task mixture and is not a deployment-invariant safety guarantee.','',
    '### Teaching, freeze, seeds, and reproducibility','',
    'The teaching collection used the unchanged v3 explainer while v4 files were being developed; its saved v3 outputs and question answers were all replayed before fitting, so development-file hash changes are not claimed to be a frozen v4 run. New teaching/calibration: layouts **70000–70019**, disturbance base **81000**, 20 episodes per condition / 100 physical trials, four goals balanced across layouts. Existing monitor residual calibration remains seeds 67400–67407 and attribute teaching seed 67420, matching v3; these are development-only. An initial frozen attempt on 69000–69019 / base 80000 was stopped and invalidated when printed inventory review found question-mark corruption from Windows shell encoding in several new Arabic strings. Its checkpoint, partial workers, logs, tests, and inventory are retained under `language_causes_v4_invalidated_69000_*`; they are excluded from all final scores. Statistical parameters were not tuned after this discovery. Explicit UTF-8 source writing and an independent ASCII-escaped Unicode regression test repair and guard the text. V4 was then frozen again before any replacement final simulation. Fresh final: **69020–69039**, disturbance base **82000**, 100 physical episodes collected by two disjoint modest GPU workers (64 planning samples each); worker files are merged only after complete nonduplicate key checks and matching calibration/source hashes. Seed ranges were checked against prior language result seed fields. Each saved direct-goal episode produces both languages; v3 and v4 replay the identical evidence. These are 100 shared physical trials, not 200 independent samples. Instruction parsing, control, and language-instruction execution were not changed or separately re-evaluated here.','',
    'Run from the repository root using the project Python:','',
    '```powershell',"$env:PYTHONPATH='.;scripts'", "$env:PYTHONIOENCODING='utf-8'", "$env:OMP_NUM_THREADS='1'",
    '.venv/Scripts/python.exe scripts/language_causes_v4_eval.py --teaching --checkpoint results/language_causes_v3_checkpoint.json --seed 70000 --count 20 --disturbance-base 81000 --output results/language_causes_v4_teaching.json',
    '.venv/Scripts/python.exe scripts/language_causes_v4_train.py',
    '.venv/Scripts/python.exe scripts/language_causes_v4_final.py',
    '.venv/Scripts/python.exe scripts/language_causes_v4_report.py --docs',
    '.venv/Scripts/python.exe scripts/language_causes_v4_tests.py --basetemp data/pytest_language_v4_utf8_inherited_20261028 --output results/language_causes_v4_tests_frozen.json',
    '```','',
    'Use a new output name to collect new experiments; `--resume` continues only incomplete matching frozen runs. Do not rerun final seeds for model selection. The report also checks source/model hashes and exactly replays all 200 bilingual explanations and all saved question answers. Tests include exhaustive controlled Arabic clause inventory matching over all causes, goal states, action frames, and representative cardinalities, plus every `none`/hedge complementizer. On this Windows installation pytest’s mode-0700 temporary directories are inaccessible; the test wrapper changes only new project test-directory creation to inherited permissions, restores `os.mkdir` afterward, and never deletes an existing directory.','',
    '### Learned absence likelihoods','',
    '| Condition | Teaching n | Silent count | P(silent given condition) | Explicit-none count | P(explicit none given condition) |',
    '|---|---:|---:|---:|---:|---:|']
    silence=checkpoint['v4']['silence']['counts'];diag=checkpoint['v4']['diagnostics']
    for c,r in silence.items():
        lines.append(f"| {c} | {r['n']} | {r['silent']} | {diag['silent']['likelihoods'][c]:.4f} | {r['explicit_none']} | {diag['explicit_none']['likelihoods'][c]:.4f} |")
    for event in ('silent','explicit_none'):
        d=diag[event];lo,hi=d['none_credible95']
        lines.extend(['',f"`{event}`: P(no disturbance given evidence) = **{d['posterior']['none']:.4f}**, posterior credible 95% [{lo:.4f}, {hi:.4f}]."])
    lines.extend(['','### Frozen final paired results','',
        'Each language has the same counts below because the evidence and semantic mappings agree. Wilson 95% intervals describe these sampled cases. Hedged reports count as issued `none` hypotheses and can still be wrong; the hedge never erases an error.','',
        '| Metric, each language | V3 | V4 |','|---|---:|---:|'])
    l=summary['languages']['english']
    for k,label in [('cause_correct','Correct cause / all episodes'),('coverage','Issued cause reports / all episodes'),('precision','Correct / issued reports'),('confident_none_precision','Correct / confident none reports'),('success','Direct-goal task success')]:
        lines.append(f"| {label} | {fmt(l['v3'][k])} | {fmt(l['v4'][k])} |")
    lines.append(f"| Hedged none reports | {l['v3']['hedged_none']} | {l['v4']['hedged_none']} |")
    p=l['paired'];lines.extend(['',f"Paired sign test, each language: only v4 correct **{p['only_v4_correct']}**, only v3 correct **{p['only_v3_correct']}**, two-sided p = **{p['sign_test_p']:.6g}**. These are identical paired outcomes in two languages, not independent replications.",'',
          '| Condition | V3 cause correct | V4 cause correct |','|---|---:|---:|'])
    for c in silence:
        lines.append(f"| {c} | {fmt(l['v3']['by_condition'][c])} | {fmt(l['v4']['by_condition'][c])} |")
    lines.extend(['','### Calibration and reliability','',
    'Primary Brier score is the binary forecast P(no disturbance) against the event-log absence of a disturbance, over every episode (lower is better). It is not a multiclass Brier score. The silent subset measures the targeted absence forecasts. A second top-label reliability analysis compares each reported confidence with correctness of the issued cause decision; abstentions are excluded from that conditional analysis and retained in the coverage score. Confident `none` precision uses a shared at-least-0.95 cutoff (and v4’s confident wording). A zero denominator is reported as not estimable, not perfect precision.','',
    '| Calibration metric, each language | V3 | V4 |','|---|---:|---:|'])
    for key,title in [('binary_none_calibration','Binary none, all episodes'),('silent_none_calibration','Binary none, silent subset'),('top_label_calibration','Top-label report correctness')]:
        a,b=l['v3'][key],l['v4'][key]
        lines.append(f"| {title}: Brier (n) | {a['brier']:.6f} ({a['n']}) | {b['brier']:.6f} ({b['n']}) |")
    lines.extend(['','| Forecast | Version | Probability bin | n | Mean stated probability | Observed frequency / accuracy [Wilson 95%] |','|---|---|---|---:|---:|---|'])
    for key,label in [('binary_none_calibration','No-disturbance forecast'),('top_label_calibration','Top-label report')]:
        for version in ('v3','v4'):
            for b in l[version][key]['bins']:
                mean='—' if b['mean_confidence'] is None else f"{b['mean_confidence']:.4f}"
                lines.append(f"| {label} | {version} | {b['bin']} | {b['n']} | {mean} | {fmt(b['observed_accuracy'])} |")
    lines.extend(['','### Complete controlled Arabic sentence inventory','',
    'The following enumerates all reachable sentence templates, plus the additional agreement examples taught to the composer (marked teaching-only). A paragraph is a composition of these sentences. `{value}` ranges over agent estimates, a nonnegative count label, or a numeric probability; infinitely many numeric spellings are represented by the template, not listed separately. No general-purpose Arabic generator or unrestricted paraphrasing is claimed. Definite noun complements retain the article; the indefinite subject after `أنّ` is taught as `قوةً خارجيةً`, and indefinite accusatives include `سببًا`, `شيئًا`, `اضطرابًا صغيرًا`, `ممكنًا`, `اعتمادًا`. Unvowelled standard spellings elsewhere do not assert fully vocalized case annotation.','',
    '| Sentence / template | Review note |','|---|---|'])
    for r in summary['arabic_inventory']:
        lines.append(f"| {r['text']} | {r['note']}"+(' (teaching-only)' if not r['runtime'] else '')+' |')
    lines.extend(['','### Unedited examples next to scorer-only truth','',
                  '| Seed / condition / event log / success | English | Arabic |','|---|---|---|'])
    selected=[]
    for c in silence:
        candidates=[r for r in summary['episodes'] if r['condition']==c]
        preferred=next((r for r in candidates if not r['belief']['posterior']),candidates[0])
        selected.append(preferred)
    for r in selected:
        truth=json.dumps(r['truth']['disturbances'],ensure_ascii=False,separators=(',',':'))
        lines.append(f"| {r['seed']} / {r['condition']} / `{truth}` / {r['truth']['success']} | {r['outputs']['v4']['english']['explanation']['text']} | {r['outputs']['v4']['arabic']['explanation']['text']} |")
    lines.extend(['','### Failures and limits','',
    'Calibration is assessed once on the new frozen set and is not guaranteed to improve every metric. The inherited non-`none` arm posteriors remain uncalibrated; false pushes and heavier/slippery confusions are outside this language-only change. A hedged “nothing unusual” is still scored wrong when a real disturbance occurred. The given balanced prior may be wrong in deployment, and the per-condition sample sizes leave broad uncertainty. Absence probabilities are conditional on the five taught conditions; unseen disturbance types and sensor failures are not calibrated here. No additional practice, controller change, monitor tuning, novel question-intent training, speech, or unrestricted Arabic morphology was performed.',''])
    for c in silence:
        bad=[r['seed'] for r in summary['episodes'] if r['condition']==c and not r['outputs']['v4']['english']['score']['cause_correct']]
        lines.append(f"- V4 cause errors/abstentions, `{c}`: {bad or 'none'}.")
    physical=[(r['condition'],r['seed']) for r in summary['episodes'] if not r['truth']['success']]
    lines.extend(['',f'Physical task failures (shared by the reporting comparisons): `{physical}`.','',
    'Artifacts: `results/language_causes_v4_teaching.json`, `_checkpoint.json`, `_final.json`, `_summary.json`, and `_tests_frozen.json`. Final summary includes all per-episode v3/v4 explanations, raw beliefs, separate truth, numeric forecasts, and reliability bins. The v4 evaluator records project-relative commands and frozen hashes; result files contain no absolute local paths.',''])
    path.write_text(original+'\n'.join(lines),encoding='utf8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='results/language_causes_v4_final.json')
    ap.add_argument('--checkpoint',default='results/language_causes_v4_checkpoint.json')
    ap.add_argument('--old',default='results/language_causes_v3_checkpoint.json')
    ap.add_argument('--output',default='results/language_causes_v4_summary.json')
    ap.add_argument('--docs',action='store_true')
    args=ap.parse_args()
    new=json.loads(Path(args.checkpoint).read_text(encoding='utf8'))
    old=json.loads(Path(args.old).read_text(encoding='utf8'))
    data=json.loads(Path(args.input).read_text(encoding='utf8'))
    if data['checkpoint_sha256']!=digest(args.checkpoint):
        raise ValueError('Frozen model changed')
    for path,hash_ in data['source_hashes'].items():
        if digest(path)!=hash_:
            raise ValueError('Frozen runtime changed: '+path)
    out=analyze(data,old,new)
    out.update(input=args.input,checkpoint=args.checkpoint,old=args.old,
               hashes={p:digest(p) for p in (args.input,args.checkpoint,args.old)},command=['python',*__import__('sys').argv])
    save(args.output,out)
    if args.docs:
        append_documentation(out,new)
    print(json.dumps(out['languages'],indent=2))
