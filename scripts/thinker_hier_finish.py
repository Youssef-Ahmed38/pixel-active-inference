"""Execute the pre-registered remaining stages without selecting on final results."""
import json,re,subprocess,sys,time
from pathlib import Path
from thinker_hier_train import SEEDS,KINDS


def call(name,args):
    print('starting',name,flush=True)
    path=Path('results/thinker_hier_stage_'+name+'.log')
    with path.open('a',encoding='utf8') as log:code=subprocess.call([sys.executable,'scripts/thinker_hier_run.py',*args],stdout=log,stderr=subprocess.STDOUT)
    path.write_text(re.sub(r'[A-Za-z]:[\\/][^\r\n\"]+','<local-path>',path.read_text(encoding='utf8')),encoding='utf8')
    if code:raise SystemExit('Failed '+name+'; inspect retained log')
    print('finished',name,flush=True)

if __name__=='__main__':
    while not all(Path(f'results/thinker_hier_{k}_s{s}_training.json').exists() for s in SEEDS for k in KINDS):time.sleep(20)
    while True:
        p=list(Path('results').glob('thinker_hier_development_manifest*.json'))
        if len(p)==2 and all(json.loads(x.read_text(encoding='utf8'))['complete'] for x in p):break
        time.sleep(20)
    call('reference_calibration',['scripts/thinker_hier_references.py','calibration'])
    call('development',['scripts/thinker_hier_evaluate.py','development'])
    call('tests',['scripts/thinker_hier_tests.py','--basetemp','data/pytest_thinker_hier_final_20261028','--output','results/thinker_hier_tests_final.json'])
    call('freeze',['scripts/thinker_hier_evaluate.py','freeze'])
    call('collection_final',['scripts/thinker_hier_collection.py','--split','final'])
    call('scoring_final',['scripts/thinker_hier_evaluate.py','final'])
    call('reference_final',['scripts/thinker_hier_references.py','final'])
    call('report',['scripts/thinker_hier_report.py'])
