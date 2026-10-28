"""Development smoke, fresh halting analysis, strict PGN audit, and regressions."""
import argparse,io,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.chess_hrm_study import load,atomic,sha,ENGINE,PLAN,GAMES
from pai.chess.adaptive_hrm import AdaptiveHRMValue,stopping_targets
from pai.chess.hrm_value import board_tensors,mirror_observation
from pai.chess.hrm_training import orbit_key
from pai.chess.distill import observation_from_fen,generate_teacher_positions
from pai.chess.uci_sync import install_sync_engine


def smoke():
    from pai.chess.ladder import balanced_openings,run_job
    root=balanced_openings(1,95400,depth=8,engine_path=ENGINE)[0]
    base=dict(kind='agent',value='runs/chess_strength_data/linear_candidate.json',
              policy='runs/chess_strength_data/policy.npz',time=.1,nodes=1000000,quiescence=2,reuse_pv=True)
    new={**base,'value':'results/chess_hrm_adaptive_model.pt'}
    rows=[]
    for color in ('white','black'):
        rows.append(run_job({'id':'smoke_'+color,'group':'development','pair':0,'candidate_color':color,
                             'white':new if color=='white' else base,'black':base if color=='white' else new,
                             'root':root,'seed':95400,'max_plies':16}))
    atomic(Path('results/chess_hrm_development_smoke.json.gz'),{'seed':95400,'games':rows,'note':'Capped smoke; excluded from strength'})
    assert all(r['failure'] is None for r in rows)
    print('Development smoke: legal commands; capped games excluded from strength',flush=True)


def halting():
    value=AdaptiveHRMValue.load('results/chess_hrm_adaptive_model.pt')
    rows=generate_teacher_positions((ENGINE,93400,400,8));seen=set(value.summary['training_positions'])
    kept=[]
    for r in rows:
        key=orbit_key(observation_from_fen(r[0]))
        if key not in seen:kept.append(r);seen.add(key)
    records=[]
    for start in range(0,len(kept),32):
        chunk=kept[start:start+32];obs=[observation_from_fen(r[0]) for r in chunk];n=len(obs)
        x,m,c=board_tensors(obs+[mirror_observation(o) for o in obs])
        with torch.no_grad():
            out=value.model.think(x,m,c);raw=out.logits['value'].squeeze(-1).numpy();steps=out.segments.numpy()
            fixed=value.model.forward_segments(x,m,c)
        base=value.baseline.predict_many(obs);y=np.asarray([r[1]*(1 if o.turn else -1) for r,o in zip(chunk,obs)])
        pred=base+500*(raw[:n]-raw[n:])/2
        predictions=np.stack([base+500*(o.logits['value'][:n,0].numpy()-o.logits['value'][n:,0].numpy())/2 for o in fixed])
        errors=np.abs(predictions-y);targets=stopping_targets(torch.from_numpy(errors),2.).numpy()
        oracle=np.array([next((s+1 for s in range(6) if targets[s,i]==0),6) for i in range(n)])
        for i,r in enumerate(chunk):
            records.append({'fen':r[0],'trajectory':r[3],'teacher_white_cp':float(y[i]),'baseline_cp':float(base[i]),
                            'adaptive_cp':float(pred[i]),'segments_views':[int(steps[i]),int(steps[i+n])],
                            'segments_mean':float((steps[i]+steps[i+n])/2),'oracle_stop_segment':int(oracle[i]),
                            'fixed_cp':predictions[:,i].tolist()})
    e=np.array([abs(r['teacher_white_cp']-r['baseline_cp']) for r in records]);s=np.array([r['segments_mean'] for r in records])
    lo,hi=np.quantile(e,[.25,.75]);easy=e<=lo;hard=e>=hi
    histogram={str(k):sum(r['segments_views'].count(k) for r in records) for k in range(1,7)}
    summary={'positions':len(records),'generated':len(rows),'mean_segments':float(s.mean()),'segments_histogram_views':histogram,
             'easy_quartile_mean_segments':float(s[easy].mean()),'hard_quartile_mean_segments':float(s[hard].mean()),
             'baseline_error_step_correlation':float(np.corrcoef(e,s)[0,1]) if s.std()>0 else None,
             'baseline_mae_cp':float(e.mean()),
             'adaptive_mae_cp':float(np.mean([abs(r['teacher_white_cp']-r['adaptive_cp']) for r in records])),
             'fixed6_mae_cp':float(np.mean([abs(r['teacher_white_cp']-r['fixed_cp'][-1]) for r in records])),
             'recurrence_saving_vs_cap':float(1-s.mean()/6)}
    atomic(Path('results/chess_hrm_final_halting.json.gz'),{'seed':93400,'model_sha256':sha('results/chess_hrm_adaptive_model.pt'),
           'summary':summary,'records':records,'note':'Teacher labels only for scoring; cap six, learned head chooses earlier stopping; not final-data tuning'})
    print(summary,flush=True)


def audit():
    from scripts.chess_strength_audit import audit_game
    plan=load(PLAN);data=load(GAMES);seen=set();points={}
    schedule={j['id']:j for j in plan['jobs']}
    for g in data['games']:
        assert g['id'] not in seen;seen.add(g['id'])
        expected=schedule[g['id']]
        assert all(g[k]==expected[k] for k in ('white','black','root','seed','candidate_color','group','pair'))
        score=audit_game(g);points.setdefault(g['group'],[]).append(score)
    assert seen==set(schedule) and data['complete']
    atomic(Path('results/chess_hrm_final_audit.json.gz'),{'passed':True,'games':len(seen),'groups':{
        k:{'games':len(v),'points':sum(s for s in v if s is not None),'unresolved':sum(s is None for s in v)} for k,v in points.items()},
        'games_sha256':sha(GAMES),'plan_sha256':sha(PLAN)})
    print('Audited all',len(seen),'PGNs',flush=True)


def tests():
    import pytest
    base=(ROOT/'data/pytest_chess_hrm_20261028').resolve()
    if base.exists():raise FileExistsError('Use a new temporary directory name')
    original=os.mkdir
    def inherited(path,mode=0o777,*a,**kw):
        target=Path(path).resolve()
        if os.name=='nt' and mode==0o700 and (target==base or target.is_relative_to(base)):mode=0o777
        return original(path,mode,*a,**kw)
    class Results:
        def __init__(self):self.rows=[]
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed:self.rows.append({'nodeid':report.nodeid,'outcome':report.outcome,'phase':report.when})
    recorder=Results();os.mkdir=inherited
    files=sorted(str(p) for pattern in ('test_chess*.py','test_thinker*.py') for p in Path('tests').glob(pattern))+['tests/test_cognition.py']
    try:status=pytest.main(files+['-q','-p','no:cacheprovider','--basetemp',str(base)],plugins=[recorder])
    finally:os.mkdir=original
    atomic(Path('results/chess_hrm_tests.json.gz'),{'exit_code':int(status),'tests':recorder.rows,'files':files,
           'command':['<project-python>','scripts/chess_hrm_evaluate.py','tests']})
    raise SystemExit(status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('smoke','halting','audit','tests'));a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode!='tests':install_sync_engine()
    globals()[a.mode]()
