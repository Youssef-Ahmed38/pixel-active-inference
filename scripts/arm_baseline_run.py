"""Execute the registered CPU/GPU experiment, with frozen final models."""
import json
from pathlib import Path
import re
import subprocess
import sys

from arm_baseline_plan import register,PREFIX,save,digest


def call(name,args):
    path=Path(PREFIX+name+'.log')
    with path.open('a',encoding='utf8') as handle:r=subprocess.run([sys.executable,*args],stdout=handle,stderr=subprocess.STDOUT)
    text=path.read_text(encoding='utf8')
    text=re.sub(r'File "([^"]+)"',lambda m:'File "'+(m.group(1).replace('\\','/').split('pixel-active-inference/')[-1] if 'pixel-active-inference' in m.group(1) else '<python-library>')+'"',text)
    path.write_text(text,encoding='utf8');save(PREFIX+'command_'+name+'.json',dict(command=['python',*args],exit_code=r.returncode))
    if r.returncode:raise RuntimeError('Stage failed; retain log and resume unchanged only: '+name)
    print('Completed',name,flush=True)


def audit():
    p=register();f=json.loads(Path(PREFIX+'freeze.json').read_text());records=[]
    for group in ('files','sources'):
        for path,sha in f[group].items():
            if digest(path)!=sha:raise ValueError('Frozen artifact changed: '+path)
    full=json.loads(Path(PREFIX+'final_full_agent.json').read_text())['records']
    from pai.baselines.statistics import paired
    for method in p['methods']:
        for seed in p['training_seeds'] if method in ('ppo','recurrent_ppo') else [None]:
            path=PREFIX+'final_'+method+(f'_s{seed}' if seed is not None else '')+'.json'
            value=json.loads(Path(path).read_text());rs=value['records']
            if not value['complete'] or len(rs)!=100 or len({r['index'] for r in rs})!=100:raise ValueError('Incomplete/duplicate evaluation')
            paired(rs,full)
            if any(r['seed']!=110000+r['index'] for r in rs):raise ValueError('Unexpected final seeds')
            if any('Ground truth:' in (r['explanation'] or '') for r in rs):raise ValueError('Truth in causal explanation')
            records.extend(rs)
    trains=[json.loads(path.read_text()) for path in Path('results').glob('arm_baselines_*_training.json')]
    if len(trains)!=6 or any(t['environment_decisions']>p['budget_steps'] for t in trains):raise ValueError('Training budget mismatch')
    if any(t['source_hashes']!=trains[0]['source_hashes'] for t in trains):raise ValueError('Training sources changed across trials')
    for path in Path('results').glob('arm_baselines_*'):
        if path.suffix in ('.json','.log') and re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/',path.read_text(encoding='utf8')):raise ValueError('Absolute local path: '+path.as_posix())
    save(PREFIX+'audit.json',dict(final_records=len(records),trained_models=len(trains),paired_layouts_and_disturbances=True,frozen_hashes_unchanged=True,training_sources_identical=True))


if __name__=='__main__':
    p=register()
    if json.loads(Path(PREFIX+'tests_final.json').read_text())['exit_code']:raise ValueError('Tests must pass')
    from concurrent.futures import ThreadPoolExecutor
    jobs=[(kind,seed) for seed in p['training_seeds'] for kind in ('ppo','recurrent_ppo')
          if not Path(PREFIX+f'{kind}_s{seed}_training.json').exists()]
    def execute(job):
        kind,seed=job
        call(f'train_{kind}_{seed}',['scripts/arm_baseline_train.py','--kind',kind,'--seed',str(seed)])
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(execute,jobs))
    if not Path(PREFIX+'calibration.json').exists():call('calibration',['scripts/arm_baseline_eval.py','calibration'])
    if not Path(PREFIX+'freeze.json').exists():call('freeze',['scripts/arm_baseline_eval.py','freeze'])
    evaluation_jobs=[(method,seed) for method in p['methods']
                     for seed in (p['training_seeds'] if method in ('ppo','recurrent_ppo') else [None])]
    def evaluate(job):
        method,seed=job;args=['scripts/arm_baseline_eval.py','evaluate','--method',method]
        if seed is not None:args+=['--seed',str(seed)]
        call(f'final_{method}_{seed}',args)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(evaluate,evaluation_jobs))
    if not Path(PREFIX+'summary.json').exists():call('report',['scripts/arm_baseline_report.py'])
    audit();print('Audit complete',flush=True)
