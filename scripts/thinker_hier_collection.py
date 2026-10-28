"""Run modest, disjoint collection workers; resumable without recollecting saved episodes."""
import argparse,subprocess,sys
from pathlib import Path
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','development','final']);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();children=[]
    if args.split=='final' and not Path('results/thinker_hier_freeze.json').exists():raise ValueError('Freeze models/readouts before final collection')
    for i in range(args.workers):
        log=Path(f'results/thinker_hier_{args.split}_worker{i}.log').open('a',encoding='utf8')
        p=subprocess.Popen([sys.executable,'scripts/thinker_hier_run.py','scripts/thinker_hier_collect.py','--split',args.split,'--workers',str(args.workers),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT)
        children.append((p,log))
    codes=[]
    for p,log in children:codes.append(p.wait());log.close()
    if any(codes):raise SystemExit('Collection failed; inspect worker log and resume unchanged')
    print('Complete:',args.split,flush=True)
