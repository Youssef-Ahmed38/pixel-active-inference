"""Run nine fixed-budget training jobs; skip only completed hashed checkpoints."""
import json,subprocess,sys,time
from pathlib import Path
from thinker_hier_train import KINDS,SEEDS
if __name__=='__main__':
    while True:
        manifests=list(Path('results').glob('thinker_hier_train_manifest*.json'))
        if len(manifests)==2 and all(json.loads(p.read_text(encoding='utf8'))['complete'] for p in manifests):break
        time.sleep(20)
    for seed in SEEDS:
        for kind in KINDS:
            prefix=f'results/thinker_hier_{kind}_s{seed}'
            if Path(prefix+'_training.json').exists():continue
            with Path(prefix+'_worker.log').open('a',encoding='utf8') as log:
                code=subprocess.call([sys.executable,'scripts/thinker_hier_run.py','scripts/thinker_hier_train.py','--kind',kind,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            if code:raise SystemExit('Training job failed: '+prefix)
            print('trained',kind,seed,flush=True)
