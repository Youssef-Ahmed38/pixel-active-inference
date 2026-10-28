"""Summarize frozen arm cause gates and append their tables to ARM_CAUSES.md."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.arm_causes import load,save,sha,KNOWN


def proportion(value):
    k=value.get('correct',0);n=value['n'];ci=value['wilson95']
    return f"{k}/{n} = {100*k/n:.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]" if n else 'not available'


def build(plan):
    lines=['\n## Frozen final results\n']
    records={}
    for tag,spec in plan['regimes'].items():
        prefix=f'results/arm_causes_{tag}'
        data=load(prefix+'_final_data.json.gz');scores=load(prefix+'_final_scores.json.gz')
        model=load(spec['model']);control=load(prefix+'_control.json.gz')
        online=load(prefix+'_online.json.gz');audit=load(prefix+'_audit.json.gz')
        native=load(prefix+'_controller_scores.json.gz')
        native_audit=load(prefix+'_controller_audit.json.gz')
        assert audit['passed']
        assert native_audit['passed']
        summaries=scores['summary'];rs=scores['rows']
        record={'summary':summaries,'paired':scores['paired'],'controller':control,
                'selected':{k:model[k]['selected'] for k in ('terminal','online')},
                'scheduled_not_fired':sum(r['scheduled_condition']!=r['truth'] for r in rs),
                'false_pushes':{},'unknown_predictions':{},'common_correct_parameter_errors':{},
                'online_prefix_replay':online['summary'],'audit':audit,
                'native_controller_diagnosis':native['summary'],'native_controller_audit':native_audit,
                'final_data_sha256':sha(prefix+'_final_data.json.gz')}
        for method in ('old','new'):
            record['false_pushes'][method]={c:sum(r['truth']==c and r[method+'_cause']=='push' for r in rs)
                                           for c in ('none','heavier_object','slippery_object')}
            record['unknown_predictions'][method]=sum(r[method+'_cause']=='unknown' for r in rs)
        for field in ('mass_abs_error_kg','offset_error_mm','onset_error_s'):
            common=[r for r in rs if field in r['old_errors'] and field in r['new_errors']]
            record['common_correct_parameter_errors'][field]={'n':len(common),**{
                method:sum(r[method+'_errors'][field] for r in common)/len(common) if common else None
                for method in ('old','new')}}
        records[tag]=record
        lo,hi=spec['seed'],spec['seed']+spec['count']-1
        lines += [f'### {tag}: seeds {lo}–{hi}\n',
                  '| Diagnosis | Old analytic | Learned |','|---|---:|---:|',
                  f"| Overall | {proportion(summaries['old'])} | {proportion(summaries['new'])} |"]
        for c in KNOWN:
            lines.append(f"| {c} recall | {proportion(summaries['old']['by_cause'][c])} | {proportion(summaries['new']['by_cause'][c])} |")
        ba={m:sum(v['recall'] for v in summaries[m]['by_cause'].values())/5 for m in ('old','new')}
        paired=scores['paired'];ci=paired['layout_bootstrap95']
        lines += [f"\nBalanced accuracy: old {100*ba['old']:.1f}%, learned {100*ba['new']:.1f}%.",
                  f"Paired gain: {100*paired['accuracy_gain']:.1f} percentage points; layout-bootstrap 95% interval "
                  f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]. New-only correct {paired['new_only_correct']}; "
                  f"old-only correct {paired['old_only_correct']}.",
                  f"Unknown predictions: {record['unknown_predictions']}. Scheduled disturbances that never fired: {record['scheduled_not_fired']}.",
                  f"False pushes by true cause: `{record['false_pushes']}`.\n",
                  'Confusion rows: none, push, heavier, slippery, camera. Columns: those five, unknown.\n',
                  '```text','old '+json.dumps(summaries['old']['confusion']),
                  'new '+json.dumps(summaries['new']['confusion']),'```\n',
                  '| Parameter, correct cause only | Old coverage / MAE | Learned coverage / MAE |',
                  '|---|---:|---:|']
        for c,field,unit in [('heavier_object','mass_abs_error_kg','kg'),('camera_shift','offset_error_mm','mm'),
                            ('push','onset_error_s','s'),('slippery_object','onset_error_s','s'),('camera_shift','onset_error_s','s')]:
            cells=[]
            for m in ('old','new'):
                v=summaries[m]['by_cause'][c]['parameters'].get(field)
                cells.append(f"{v['available_n']}/{v['correct_cause_n']} / {v['mean']:.4f} {unit}; "
                             f"{v['within_tolerance']} within tolerance" if v else 'no estimates')
            lines.append(f"| {c}: {field} | {' | '.join(cells)} |")
        lines+=['\nCoverage denominators are correctly diagnosed cases, not all injected cases. '
                'Errors across methods can involve different cases. Matched-case MAEs: '
                f"`{record['common_correct_parameter_errors']}`.\n",
                '#### Actual closed-loop controller subset\n',
                f"Seeds {lo}–{lo+spec['controller_count']-1}, frozen prefix models; task goal success:\n",
                '| Condition | Analytic controller | Learned controller |','|---|---:|---:|']
        for c in (*KNOWN,'all'):
            v=control['summary'][c]
            lines.append(f"| {c} | {proportion(v['baseline'])} | {proportion(v['learned'])} |")
        ci=control['paired']['layout_bootstrap95']
        lines += [f"\nTask-success difference layout-bootstrap interval: [{100*ci[0]:.1f}, {100*ci[1]:.1f}] points. "
                  f"Learned-only successes {control['paired']['learned_only_success']}; "
                  f"baseline-only successes {control['paired']['baseline_only_success']}.",
                  'This five-layout controller subset is exploratory. A zero-width bootstrap interval means '
                  'all observed paired differences were zero; it does not establish zero effect on unseen layouts.',
                  f"Terminal diagnosis on the learned controller's own trajectories: "
                  f"learned {proportion(native['summary']['new'])}; analytic post-hoc comparator "
                  f"{proportion(native['summary']['old'])}. This is separate from the main baseline-trajectory table.\n",
                  '#### Held-out evidence examples and failures\n',
                  '| Seed / scheduled condition | Event-log cause (scoring only) | Old | Learned (confidence) | Goal |',
                  '|---|---|---|---|---:|']
        # Include improvements and failures, then representative correct cases.
        chosen=[]
        for predicate,limit in [(lambda r:r['new_cause']!=r['truth'],5),
                                (lambda r:r['new_cause']==r['truth'] and r['old_cause']!=r['truth'],5),
                                (lambda r:True,3)]:
            count=0
            for r in rs:
                if predicate(r) and r not in chosen:
                    chosen.append(r);count+=1
                    if count>=limit:break
        for r in chosen:
            lines.append(f"| {r['seed']} / {r['scheduled_condition']} | {r['truth']} | {r['old_cause']} | "
                         f"{r['new_cause']} ({max(r['new_posterior'].values()):.3f}) | {r['success']} |")
        selected=record['selected']
        lines += ['\nDevelopment selection (layout-held-out; not final):\n',
                  '| Model | Features / shrinkage / family / prior | Accuracy | Balanced accuracy |',
                  '|---|---|---:|---:|']
        for kind,v in selected.items():
            lines.append(f"| {kind} | {v['feature_set']} / {v['shrinkage']} / {v['family']} / {v['prior_kind']} | "
                         f"{100*v['score']['accuracy']:.1f}% | {100*v['score']['balanced_accuracy']:.1f}% |")
        lines += [f"\nTerminal / prefix training records: {model['training_counts']}.\n",
                  '| Prefix replay on final factual trace | Old accuracy | Learned accuracy |',
                  '|---|---:|---:|']
        for stage,methods in online['summary'].items():
            lines.append(f"| {stage} | {methods['old']['correct']}/{methods['old']['n']} | "
                         f"{methods['new']['correct']}/{methods['new']['n']} |")
        lines.append('')
    return records,'\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('results/arm_causes_plan.json'))
    parser.add_argument('--output',type=Path,default=Path('results/arm_causes_summary.json'))
    parser.add_argument('--doc',type=Path,default=Path('docs/ARM_CAUSES.md'))
    args=parser.parse_args()
    records,text=build(load(args.plan))
    previous=args.doc.read_text(encoding='utf-8')
    if '## Frozen final results' in previous:raise ValueError('Final tables already present')
    save(args.output,{'plan_sha256':sha(args.plan),'regimes':records})
    args.doc.write_text(previous+text+'\n',encoding='utf-8')
