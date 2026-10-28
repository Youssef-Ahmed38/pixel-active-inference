"""Two modest disjoint final workers; checkpoint and merge exactly 100 episodes."""
import json
from pathlib import Path
import subprocess
import sys
from language_causes_v4_train import save,digest

GROUPS=[['none','push','heavier_object'],['slippery_object','camera_shift']]

if __name__=='__main__':
    checkpoint=Path('results/language_causes_v4_checkpoint.json')
    final=Path('results/language_causes_v4_final.json')
    if final.exists():
        raise FileExistsError('Final already merged; use report, never rerun frozen final')
    children=[]
    for i,conditions in enumerate(GROUPS):
        output=Path(f'results/language_causes_v4_final_worker{i}.json')
        if output.exists() and json.loads(output.read_text(encoding='utf8'))['complete']:
            continue
        args=[sys.executable,'scripts/language_causes_v4_eval.py','--baseline-only','--checkpoint',str(checkpoint),
              '--seed','69020','--count','20','--disturbance-base','82000','--conditions',*conditions,'--output',str(output)]
        if output.exists():args.append('--resume')
        log=Path(f'results/language_causes_v4_final_worker{i}.log')
        handle=log.open('a',encoding='utf8')
        process=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT)
        children.append((i,process,handle))
    codes=[]
    for i,process,handle in children:
        code=process.wait();handle.close();codes.append(code)
        import re
        log=Path(f'results/language_causes_v4_final_worker{i}.log')
        log.write_text(re.sub(r'[A-Za-z]:[\\/][^\r\n\"]+', '<local-path>', log.read_text(encoding='utf8')),encoding='utf8')
        print('worker',i,'exit',code,flush=True)
    if any(codes):
        raise SystemExit('An incomplete worker failed; inspect its log and resume without changing the frozen configuration')
    reports=[json.loads(Path(f'results/language_causes_v4_final_worker{i}.json').read_text(encoding='utf8')) for i in range(2)]
    for field in ('seeds','checkpoint_sha256','config','source_hashes','world_model_sha256','calibration','question_wording'):
        if reports[0][field]!=reports[1][field]:raise ValueError('Workers disagree: '+field)
    all_rows=[row for r in reports for row in r['episodes']]
    expected={(s,c,'baseline') for s in range(69020,69040) for group in GROUPS for c in group}
    keys=[(r['seed'],r['condition'],r['mode']) for r in all_rows]
    if len(keys)!=100 or set(keys)!=expected:raise ValueError('Duplicate or missing final cases')
    merged=dict(reports[0],conditions=[c for group in GROUPS for c in group],episodes=all_rows,
                command=['python','scripts/language_causes_v4_final.py'],
                workers={f'results/language_causes_v4_final_worker{i}.json':digest(f'results/language_causes_v4_final_worker{i}.json') for i in range(2)})
    save(final,merged)
    print('Merged 100 distinct final episodes; no simulation repeated',flush=True)
