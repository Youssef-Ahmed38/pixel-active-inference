"""Frozen v2 evaluation: anonymous grounding, varied goals, fresh question wording."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch

from pai.agents.slice_agent import run_episode
from pai.causes.inference import calibrate
from pai.causes.credit import assign_phase_credit
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.envs.predicates import perceived_predicates
from pai.goals.relations import RelationalGoal
from pai.cognition.adapters import arm_state
from pai.language.association import BeliefCoordinates,EvidenceLexicon,QuestionIntent,AssociatedExplainer,SilenceModel
from pai.language.association_teacher import FINAL_QUESTIONS,truth_cause
from pai.language.causal import EpisodeEvidence
from pai.language.causal_teacher import PHRASES
from pai.language.grounding import GroundedLanguage,GroundedFrame
from pai.language.observation import LearnedAttributes
from pai.language.teacher import curriculum,utterance
from pai.perception.camera_frame import CameraFramePerception
from pai.world.entities import entity_names
from pai.world.model import load_world_model
from slice_eval import disturbance_for,CONDITIONS

GOALS=[GroundedFrame('instruction','on','red','block','blue','plate'),
       GroundedFrame('instruction','in','yellow','block','green','bowl'),
       GroundedFrame('instruction','on','blue','block','blue','plate'),
       GroundedFrame('instruction','in','red','block','green','bowl')]


def signature(frame):
    return (frame.predicate,frame.subject_color,frame.subject_kind,frame.target_color,frame.target_kind)


def instruction_model(language):
    held={signature(f) for f in GOALS}
    rows=[r for r in curriculum(language)[0] if signature(r[1]) not in held]
    model=GroundedLanguage.fit(rows)
    for frame in GOALS:
        assert all(signature(f)!=signature(frame) for _,f in rows)
        assert model.parse(utterance(frame,language))==frame
    return model,len(rows)


def load_explainers(checkpoint):
    coordinates=BeliefCoordinates(**checkpoint['coordinates'])
    silence=None
    if 'silence' in checkpoint:
        silence=SilenceModel(**{k:v for k,v in checkpoint['silence'].items() if k!='p_none_given_silent'})
    return {lang:AssociatedExplainer(coordinates,EvidenceLexicon(**m['lexicon']),m['templates'],
                                     QuestionIntent(**m['questions']),silence=silence)
            for lang,m in checkpoint['models'].items()}


def score_v2(output,truth,language,plan_dt=.1,control_dt=.02):
    """Teacher semantics appear only in the scorer, never the runtime learner."""
    decoder={v:k[6:] for k,v in PHRASES[language].items() if k.startswith('cause:')}
    reported=decoder.get(output['phrase'])
    events=truth['disturbances']
    event_causes=[dict(payload='heavier_object',friction='slippery_object').get(e['disturbance'],e['disturbance']) for e in events]
    correct=reported in (event_causes or ['none'])
    matches=[e for e,c in zip(events,event_causes) if c==output['agent_cause']]
    params=[]
    for claim in output['numeric_claims']:
        target=None
        semantic=None
        if claim['template']==PHRASES[language]['estimate:extra_mass_kg']:
            semantic='mass_kg'
            if len(matches)==1 and 'mass' in matches[0]['params']:
                target=[matches[0]['params']['mass']]
            tolerance=.1
        elif claim['template']==PHRASES[language]['estimate:camera_offset_mm']:
            semantic='offset_mm'
            if len(matches)==1 and 'offset' in matches[0]['params']:
                target=(1000*np.array(matches[0]['params']['offset'])).tolist()
            tolerance=10.
        elif claim['template']==PHRASES[language]['estimate:onset']:
            semantic='onset_s'
            if len(matches)==1:
                target=[matches[0]['t']*control_dt]
            tolerance=.3
        else:
            raise ValueError('Unknown taught numeric phrase')
        value=np.asarray(claim['value'])* (plan_dt if semantic=='onset_s' else 1.)
        error=float(np.linalg.norm(value-np.asarray(target))) if target is not None and len(value)==len(target) else None
        params.append(dict(semantic=semantic,event_matched=target is not None,error=error,tolerance=tolerance,
                           passed=bool(correct and error is not None and error<=tolerance)))
    return dict(reported_cause=reported,truth_causes=event_causes or ['none'],cause_correct=correct,
                goal_correct=output['goal_observed']==truth['success'],parameters=params)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=67000)
    ap.add_argument('--count',type=int,default=10)
    ap.add_argument('--samples',type=int,default=64)
    ap.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    ap.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--teaching','--baseline-only',action='store_true',help='Collect one direct-goal physical trial per case, with bilingual post-episode reports')
    ap.add_argument('--disturbance-base',type=int,default=78000,
                    help='disturbance draws: base + 100*condition + layout index (v2 final used 78000)')
    args=ap.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    checkpoint=json.loads(args.checkpoint.read_text(encoding='utf-8'))
    if set(range(args.seed,args.seed+args.count)) & set(checkpoint['development_seeds']):
        raise ValueError('Final seed overlaps teaching')
    torch.set_num_threads(1)
    cfg=load_config('configs/tabletop.yaml',['env.render_images=false',f'slice.samples={args.samples}'])
    cfg.env.plate.rgba=[.15,.3,.85,1]
    cfg.env.bowl.rgba=[.15,.7,.2,1]
    if checkpoint.get('v4'):
        if not args.teaching:
            raise ValueError('V4 final plan uses --baseline-only')
        plan=checkpoint['v4']['final_plan']
        if list(range(args.seed,args.seed+args.count))!=plan['seeds'] or args.disturbance_base!=plan['disturbance_base']:
            raise ValueError('Final seeds differ from frozen plan')
        for filename,hash_ in checkpoint['v4']['frozen_runtime'].items():
            if digest(filename)!=hash_:
                raise ValueError('Frozen runtime changed: '+filename)
    explainers=load_explainers(checkpoint) if not checkpoint.get('v4') else __import__('pai.language.causal_v4',fromlist=['load_v4']).load_v4(checkpoint)
    models={lang:instruction_model(lang) for lang in explainers}
    hashes={str(p):digest(p) for pattern in ('pai/language/*.py','scripts/language_causes_v4*.py') for p in Path('.').glob(pattern)}
    for filename in ('pai/agents/slice_agent.py','pai/causes/inference.py','pai/causes/credit.py','pai/envs/predicates.py'):
        hashes[filename]=digest(filename)
    report=dict(version=4,conditions=args.conditions,seeds=list(range(args.seed,args.seed+args.count)),count=args.count,
                checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),config=cfg.to_dict(),
                device=args.device,world_model_sha256=digest(cfg.slice.world_model),source_hashes=hashes,
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                command=['python',*sys.argv],attribute_seed=67420,calibration_seeds=list(range(67400,67408)),
                instruction_training={k:n for k,(_,n) in models.items()},heldout_goals=[asdict(f) for f in GOALS],
                question_wording={lang:[dict(text=t,expected=i,predicted=ex.questions.predict(t))
                                       for t,i in FINAL_QUESTIONS[lang]] for lang,ex in explainers.items()},
                episodes=[],complete=False)
    if args.resume:
        old=json.loads(args.output.read_text(encoding='utf-8'))
        for key in ('seeds','conditions','checkpoint_sha256','config','source_hashes','world_model_sha256'):
            if old[key]!=report[key]:
                raise ValueError('Cannot resume different frozen configuration: '+key)
        if old['complete']:
            raise ValueError('Already completed')
        report['episodes']=old['episodes']
    def save():
        temporary=args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(args.output)
    wm=load_world_model(cfg.slice.world_model,args.device)
    env=TabletopEnv(cfg.env,disturbances=[])
    names=entity_names(env)
    sense=CameraFramePerception(env)
    tokens=sense(env.reset(seed=67420))
    attributes=LearnedAttributes.fit([(token,dict(kind='gripper' if name=='gripper' else env.objects[name].kind,
        **({} if name=='gripper' else dict(color='blue' if name=='plate' else 'green' if name=='bowl' else name))))
                                      for name,token in zip(names,tokens)])
    raw=[]
    for seed in report['calibration_seeds']:
        run_episode(env,wm,cfg,-1,seed,raw=raw,device=args.device)
        print('calibration',seed,flush=True)
    calibration=calibrate(np.stack([r[0] for r in raw]),np.stack([r[1] for r in raw]),[r[2] for r in raw])
    report['calibration']=dict(floor=calibration.floor.tolist(),threshold=calibration.threshold,
        step_floors={k:v.tolist() for k,v in calibration.step_floors.items()},step_thresholds=calibration.step_thresholds)
    save()
    for ci,condition in enumerate(CONDITIONS):
        if condition not in args.conditions:
            continue
        for i,seed in enumerate(report['seeds']):
            frame=GOALS[i%len(GOALS)]
            expected=(frame.predicate,frame.subject_color,'plate' if frame.target_kind=='plate' else 'bowl')
            disturbance_seed=args.disturbance_base+ci*100+i
            disturbances=disturbance_for(condition,np.random.default_rng(disturbance_seed))
            for d in disturbances:
                if d.get('body')=='red':
                    d['body']=expected[1]
            for mode in (('baseline',) if args.teaching else ('baseline','english','arabic')):
                if any((r['condition'],r['seed'],r['mode'])==(condition,seed,mode) for r in report['episodes']):
                    continue
                relation,obj,target=expected
                text=None
                if mode!='baseline':
                    obs=env.reset(seed=seed)
                    sense.reset()
                    initial=sense(obs)
                    belief=attributes.enrich(arm_state(initial,names,0),initial,names)
                    text=utterance(frame,mode)
                    goal=models[mode][0].goal(text,belief)
                    relation,obj,target=goal.predicate,goal.subject,goal.object
                rec=run_episode(env,wm,cfg,seed*100+ci*3+('baseline','english','arabic').index(mode),seed,
                    disturbances,relation,obj,target,device=args.device,calibration=calibration,perceive=sense)
                final=sense(env._observe())
                observed=(relation,obj,target) in perceived_predicates(env.objects,final,names)
                subgoals=RelationalGoal(relation,obj,target,names,env.objects).subgoals
                credits=[asdict(c) for c in assign_phase_credit(wm,rec.tokens,rec.actions,rec.phases,subgoals,device=args.device)]
                evidence=EpisodeEvidence.from_record(rec,goal_observed=observed,credits=credits)
                outputs={}
                for lang,ex in explainers.items():
                    explanation=ex.explain(evidence)
                    qa=[dict(text=q,expected=intent,predicted=ex.questions.predict(q),answer=ex.answer(q,evidence))
                        for q,intent in FINAL_QUESTIONS[lang]]
                    outputs[lang]=dict(explanation=explanation,qa=qa)
                # Only after all language decisions: expose truth to the scorer.
                truth=dict(success=rec.success,disturbances=rec.true_disturbances,events=rec.events)
                for lang,o in outputs.items():
                    o['score']=score_v2(o['explanation'],truth,lang) if not checkpoint.get('v4') else __import__('language_causes_v4_score',fromlist=['score_v4']).score_v4(o['explanation'],truth,lang)
                report['episodes'].append(dict(seed=seed,condition=condition,mode=mode,goal=list(expected),
                    disturbance_seed=disturbance_seed,instruction=text,resolved_goal=[relation,obj,target],steps=rec.steps,
                    belief=asdict(evidence),outputs=outputs,truth=truth))
                save()
                print(condition,seed,mode,expected,'success',rec.success,'phrase',outputs['english']['score']['reported_cause'],flush=True)
    report['complete']=True
    save()


if __name__=='__main__':
    main()
