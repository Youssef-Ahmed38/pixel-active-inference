"""Checkpointed, frozen equal-time HRM match and enlarged Stockfish ladder.

All persisted paths are project-relative. The engine is an opponent/teacher only.
"""
import argparse,gzip,hashlib,json,os,re,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from pai.chess.ladder import (balanced_openings,run_job,pair_statistics,anchored_elo)
from pai.chess.hrm_value import load_value
from scripts.chess_hrm_train import save,sha
from pai.chess.uci_sync import install_sync_engine

PLAN=Path('results/chess_hrm_search_plan.json.gz')
GAMES=Path('results/chess_hrm_search_games.json.gz')
ENGINE='runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe'


def load(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def fingerprint():
    paths=[*Path('pai/chess').glob('*.py'),*Path('pai/thinker').glob('*.py'),
           *(p for p in Path('scripts').glob('chess_hrm_*.py') if p.name!='chess_hrm_report.py')]
    return {p.as_posix():sha(p) for p in sorted(paths)}


def safe(data):
    """Reject absolute path disclosure rather than silently persisting it."""
    blob=json.dumps(data,ensure_ascii=False)
    if re.search(r'[A-Za-z]:[\\/]',blob) or '/Users/' in blob or '/home/' in blob:
        raise ValueError('absolute local path in result')
    return data


def atomic(path,data):
    tmp=Path(str(path)+'.tmp.gz');save(tmp,safe(data));os.replace(tmp,path)


def register(seconds=.2,pairs=20):
    if PLAN.exists():raise FileExistsError(PLAN)
    if pairs<=10:raise ValueError('more opening pairs than the old ten required')
    base=json.loads(Path('runs/chess_strength_data/promoted_agent.json').read_text())['config']
    base={k:v.replace('\\','/') if isinstance(v,str) else v for k,v in base.items()}
    base.update(time=seconds,nodes=1000000)
    candidate={**base,'value':'results/chess_hrm_adaptive_model.pt'}
    keys=set()
    for c in (base,candidate):keys.update(load_value(c['value']).summary.get('training_positions',()))
    from pai.chess.policy import LearnedPolicyPrior
    keys.update(LearnedPolicyPrior.load(base['policy']).summary.get('training_positions',()))
    from pai.chess.distill import observation_from_fen
    from pai.chess.hrm_training import orbit_key
    for path in Path('results').glob('chess*.meta.json'):
        old=json.loads(path.read_text())
        for fen in old.get('roots',[]):keys.add(orbit_key(observation_from_fen(fen)))
    books={}
    for name,count,seed in [('paired',10,97400),('ladder',pairs,98400)]:
        books[name]=balanced_openings(count,seed,depth=8,engine_path=ENGINE,excluded_keys=keys)
        keys.update(orbit_key(observation_from_fen(fen)) for fen in books[name])
    hashes={p:sha(p) for p in [base['value'],candidate['value'],base['policy'],ENGINE,
                              'runs/chess_rules/rules.json','runs/chess_rules/state.json','runs/chess_rules/endings.json']}
    jobs=[]
    def add(group,player,opponent,roots,seed):
        for pair,root in enumerate(roots):
            for color in ('white','black'):
                jobs.append({'id':f'{group}|{pair}|{color}','group':group,'pair':pair,'root':root,
                             'candidate_color':color,'white':player if color=='white' else opponent,
                             'black':opponent if color=='white' else player,'seed':seed+pair,'max_plies':1024})
    add('paired',candidate,base,books['paired'],97400)
    for tag,player in [('baseline',base),('adaptive',candidate)]:
        for elo in (1320,1500,1700):
            add(f'{tag}_elo{elo}',player,{'kind':'engine','uci_elo':elo,'movetime':seconds,'engine_path':ENGINE},books['ladder'],98400)
    plan={'seeds':{'training':91400,'development_openings':95400,'final_paired_openings':97400,
                    'final_ladder_openings':98400,'final_positions':93400},'seconds_per_move':seconds,
          'pairs_per_ladder_rung':pairs,'paired_match_pairs':10,'books':books,'jobs':jobs,
          'sources':fingerprint(),'checkpoint_hashes':hashes,
          'given':'soft equal movetime, million-node safety cap, quiescence 2, same learned prior, 1024-ply ceiling',
          'rule':'No final data for fitting or selection; incomplete games never scored as draws',
          'command':['<project-python>','scripts/chess_hrm_study.py','register','--seconds',str(seconds),'--pairs',str(pairs)]}
    atomic(PLAN,plan);print('Registered',len(jobs),'games;',pairs*2,'games/rung/player;',seconds,'seconds/move',flush=True)


def run(workers):
    plan=load(PLAN)
    if plan['sources']!=fingerprint() or any(sha(p)!=v for p,v in plan['checkpoint_hashes'].items()):
        raise ValueError('Frozen code/checkpoint mismatch')
    data=load(GAMES) if GAMES.exists() else {'plan_sha256':sha(PLAN),'games':[],'complete':False}
    assert data['plan_sha256']==sha(PLAN)
    done={g['id'] for g in data['games']};pending=[j for j in plan['jobs'] if j['id'] not in done]
    print('Retaining',len(done),'games; remaining',len(pending),flush=True)
    if pending:
        processes=[subprocess.Popen([sys.executable,'scripts/chess_hrm_study.py','worker','--workers',str(workers),'--index',str(i)],
                       stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) for i in range(workers)]
        while True:
            for i in range(workers):
                path=Path(f'results/chess_hrm_worker_{i}.json.gz')
                if not path.exists():continue
                saved=load(path)
                assert saved['plan_sha256']==sha(PLAN)
                for g in saved['games']:
                    if g['id'] in done:continue
                    data['games'].append(g);done.add(g['id']);atomic(GAMES,data)
                    print(len(data['games']),'/',len(plan['jobs']),g['id'],g['status'],g['result'],g['plies'],round(g['wall_seconds']),flush=True)
            if all(p.poll() is not None for p in processes):break
            time.sleep(1)
        for process in processes:
            _,err=process.communicate()
            if process.returncode:
                # Error goes to the interactive console, never to a result file.
                print(err.decode(errors='replace'),flush=True)
                raise RuntimeError('evaluation worker failed; completed games retained')
        # A worker can finish between the last file read and poll(). Drain once
        # more after every process has exited, preserving its final checkpoint.
        for i in range(workers):
            path=Path(f'results/chess_hrm_worker_{i}.json.gz')
            if not path.exists():continue
            saved=load(path);assert saved['plan_sha256']==sha(PLAN)
            for g in saved['games']:
                if g['id'] not in done:data['games'].append(g);done.add(g['id'])
    data['complete']=len(data['games'])==len(plan['jobs']);atomic(GAMES,data)


def worker(index,workers):
    """Own player/engine cache per process; no named-pipe multiprocessing API."""
    import msvcrt
    lease=Path(f'results/chess_hrm_worker_{index}.lock')
    with lease.open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        plan=load(PLAN)
        assert plan['sources']==fingerprint()
        path=Path(f'results/chess_hrm_worker_{index}.json.gz')
        data=load(path) if path.exists() else {'plan_sha256':sha(PLAN),'games':[]}
        assert data['plan_sha256']==sha(PLAN)
        done={g['id'] for g in data['games']}
        if GAMES.exists():done.update(g['id'] for g in load(GAMES)['games'])
        for j in plan['jobs'][index::workers]:
            if j['id'] in done:continue
            data['games'].append(run_job(j));atomic(path,data)


def analyse():
    plan=load(PLAN);data=load(GAMES);games=data['games'];groups={}
    for group in dict.fromkeys(j['group'] for j in plan['jobs']):
        rows=[r for r in games if r['group']==group]
        groups[group]=pair_statistics(rows)
        groups[group]['scheduled_games']=sum(j['group']==group for j in plan['jobs'])
        stats=[s for g in rows for s in g['search_records']]
        groups[group]['mean_search_seconds']=float(np.mean([s['seconds'] for s in stats])) if stats else None
        groups[group]['maximum_search_seconds']=max((s['seconds'] for s in stats),default=None)
        groups[group]['mean_nodes']=float(np.mean([s['nodes'] for s in stats])) if stats else None
    ratings={tag:anchored_elo([(elo,[r for r in games if r['group']==f'{tag}_elo{elo}']) for elo in (1320,1500,1700)])
             for tag in ('baseline','adaptive')}
    report={'plan_sha256':sha(PLAN),'games_sha256':sha(GAMES),'schedule_complete':data['complete'],
            'groups':groups,'conditional_engine_anchor_elo':ratings,
            'limitations':['Conditional short-time engine rating; not human Elo','Soft deadlines and CPU contention',
                           'Unresolved games excluded, not draws','No promotion inferred from value error alone']}
    atomic(Path('results/chess_hrm_search_summary.json.gz'),report)
    print(json.dumps({'paired':groups.get('paired'),'elo':ratings},indent=2),flush=True)


if __name__=='__main__':
    install_sync_engine()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run','analyse','worker'))
    p.add_argument('--seconds',type=float,default=.2);p.add_argument('--pairs',type=int,default=20)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.mode=='register':register(a.seconds,a.pairs)
    elif a.mode=='run':run(a.workers)
    elif a.mode=='worker':worker(a.index,a.workers)
    else:analyse()
