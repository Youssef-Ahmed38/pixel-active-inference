"""Fresh paired arm evaluation; truth is accessed only after utterances are made."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import mujoco

from pai.agents.slice_agent import run_episode
from pai.causes.inference import calibrate
from pai.causes.credit import assign_phase_credit
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.envs.predicates import perceived_predicates
from pai.goals.relations import RelationalGoal
from pai.cognition.adapters import arm_state
from pai.language.causal import ClauseLanguage, CausalExplainer, EpisodeEvidence
from pai.language.causal_teacher import teaching_rows, episode_teaching_rows
from pai.language.grounding import GroundedLanguage, GroundedFrame
from pai.language.observation import LearnedAttributes
from pai.language.teacher import curriculum, utterance
from pai.perception.camera_frame import CameraFramePerception
from pai.world.entities import entity_names
from pai.world.model import load_world_model
from slice_eval import disturbance_for, CONDITIONS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(explanation, record, plan_dt):
    mapping = {'payload': 'heavier_object', 'friction': 'slippery_object'}
    truth = record.true_disturbances
    causes = {mapping.get(e['disturbance'], e['disturbance']) for e in truth} or {'none'}
    correct = explanation['cause'] in causes
    checks = {}
    for event in truth:
        prm, est = event['params'], explanation['estimates']
        if 'extra_mass_kg' in est and 'mass' in prm:
            checks['extra_mass_kg'] = dict(error=abs(est['extra_mass_kg']-prm['mass']), tolerance=.1)
        if 'camera_offset_mm' in est and 'offset' in prm:
            checks['camera_offset_mm'] = dict(error=float(np.linalg.norm(np.array(est['camera_offset_mm'])-1000*np.array(prm['offset']))), tolerance=10.)
        if 'onset' in est:
            checks['onset'] = dict(error=abs(est['onset']*plan_dt-event['t']*.02), tolerance=.3)
    for check in checks.values():
        check['passed'] = bool(correct and check['error'] <= check['tolerance'])
    return dict(cause_correct=correct, truth_causes=sorted(causes), parameters=checks,
                goal_correct=explanation['goal_observed'] == record.success)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--count', type=int, default=2, help='layouts per condition per mode')
    ap.add_argument('--split', choices=['development', 'final'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--teaching', type=Path, help='Completed development episodes for supervised cause phrases')
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--resume', action='store_true', help='Continue an interrupted development run only')
    args = ap.parse_args()
    previous = None
    if args.resume:
        if args.split != 'development':
            raise ValueError('Final evaluation cannot mix code revisions via resume')
        previous = json.loads(args.output.read_text(encoding='utf-8'))
        if previous['seeds'] != list(range(args.seed,args.seed+args.count)) or previous['complete']:
            raise ValueError('Resume requires matching incomplete development run')
    elif args.output.exists():
        raise FileExistsError('Preserve prior evaluations; choose a new output')
    torch.set_num_threads(1)
    cfg = load_config('configs/tabletop.yaml', ['env.render_images=false'])
    cfg.env.plate.rgba = [.15, .3, .85, 1]
    cfg.env.bowl.rgba = [.15, .7, .2, 1]
    device = args.device
    wm = load_world_model(cfg.slice.world_model, device)
    env = TabletopEnv(cfg.env, disturbances=[])
    names = entity_names(env)
    # Attribute supervision from an independent clean teaching scene.
    sense = CameraFramePerception(env)
    tokens = sense(env.reset(seed=58000))
    attr_rows = []
    for name, token in zip(names, tokens):
        labels = {'kind': 'gripper' if name == 'gripper' else env.objects[name].kind}
        if name != 'gripper':
            labels['color'] = 'blue' if name == 'plate' else 'green' if name == 'bowl' else name
        attr_rows.append((token, labels))
    attributes = LearnedAttributes.fit(attr_rows)
    teaching = json.loads(args.teaching.read_text(encoding='utf-8')) if args.teaching else None
    if args.split == 'final' and (not teaching or not teaching['complete'] or teaching['split'] != 'development'):
        raise ValueError('Final evaluation requires completed development teaching episodes')
    if teaching and set(teaching['seeds']) & set(range(args.seed,args.seed+args.count)):
        raise ValueError('Teaching/evaluation seeds overlap')
    teaching_data = {lang: episode_teaching_rows(lang,teaching['episodes']) if teaching else teaching_rows(lang)
                     for lang in ('english','arabic')}
    languages = {lang: ClauseLanguage.fit(rows) for lang,rows in teaching_data.items()}
    if args.split == 'final' and any('cause:'+c not in model.templates
                                    for model in languages.values() for c in CONDITIONS):
        raise ValueError('Development teaching must contain all five observed cause classes')
    instruction_models = {lang: GroundedLanguage.fit(curriculum(lang)[0]) for lang in languages}
    # Fixed, disjoint calibration set; never fit on final observations.
    raw = []
    for seed in range(58010, 58018):
        run_episode(env, wm, cfg, -1, seed, raw=raw, device=device)
        print('calibration', seed, flush=True)
    calibration = calibrate(np.stack([r[0] for r in raw]), np.stack([r[1] for r in raw]), [r[2] for r in raw])
    report = dict(split=args.split, device=device, seeds=[args.seed+i for i in range(args.count)],
                  calibration_seeds=list(range(58010,58018)), attribute_seed=58000,
                  command=subprocess.list2cmdline(sys.argv), commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  code_hashes={str(p):digest(p) for pattern in ('pai/language/*.py','scripts/language_causes.py') for p in Path('.').glob(pattern)},
                  world_model_sha256=digest(cfg.slice.world_model), config=cfg.to_dict(),
                  threshold=.9, parameter_tolerances={'mass_kg':.1,'camera_vector_mm':10,'onset_seconds':.3},
                  learned_templates={k:v.templates for k,v in languages.items()}, episodes=[], complete=False)
    report['runtime'] = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                             numpy=np.__version__, mujoco=mujoco.__version__)
    report['calibration'] = dict(floor=calibration.floor.tolist(), threshold=calibration.threshold,
             step_floors={k:v.tolist() for k,v in calibration.step_floors.items()},
             step_thresholds=calibration.step_thresholds)
    for filename in ('pai/agents/slice_agent.py','pai/causes/inference.py','pai/causes/credit.py',
                     'pai/perception/camera_frame.py','pai/envs/predicates.py','scripts/slice_eval.py'):
        report['code_hashes'][filename] = digest(filename)
    report['teaching'] = dict(path=str(args.teaching), sha256=digest(args.teaching),
                             seeds=teaching['seeds']) if teaching else {'source':'synthetic development curriculum'}
    report['teaching_rows'] = teaching_data
    if previous:
        report['episodes'] = previous['episodes']
        report['resumed_from'] = {k:v for k,v in previous.items() if k not in ('episodes','teaching_rows','learned_templates')}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save()
    for ci, condition in enumerate(CONDITIONS):
        for i in range(args.count):
            seed = args.seed+i
            disturbance_seed = args.seed+10000+ci*100+i
            disturbances = disturbance_for(condition, np.random.default_rng(disturbance_seed))
            for mode in ('baseline','english','arabic'):
                if any(r['seed']==seed and r['condition']==condition and r['mode']==mode for r in report['episodes']):
                    continue
                relation, obj, target = 'on','red','plate'
                instruction = None
                if mode != 'baseline':
                    obs = env.reset(seed=seed)
                    sense.reset()
                    initial = sense(obs)
                    belief = attributes.enrich(arm_state(initial,names,0), initial,names)
                    instruction = utterance(GroundedFrame('instruction','on','red','block','blue','plate'),mode)
                    goal = instruction_models[mode].goal(instruction, belief)
                    relation,obj,target = goal.predicate,goal.subject,goal.object
                # Reset perception inside run_episode; its final correction remains accessible.
                rec = run_episode(env,wm,cfg,args.seed*100+len(report['episodes']),seed,disturbances,
                                  relation,obj,target,device=device,calibration=calibration,perceive=sense)
                # Fresh final perception after settling, with agent-estimated camera correction.
                final = sense(env._observe())
                observed = (relation,obj,target) in perceived_predicates(env.objects,final,names)
                subgoals = RelationalGoal(relation,obj,target,names,env.objects).subgoals
                credits = [asdict(r) for r in assign_phase_credit(wm,rec.tokens,rec.actions,rec.phases,subgoals,device=device)]
                evidence = EpisodeEvidence.from_record(rec,goal_observed=observed,credits=credits)
                outputs = {}
                for lang, model in languages.items():
                    explainer = CausalExplainer(model)
                    explanation = explainer.explain(evidence)
                    questions = {key: model.say('q:'+key+(':episode' if teaching else ''),episode=rec.episode)
                                 for key in ('why','drop','cf:remove_push','cf:open_early')}
                    answers = {key:explainer.answer(text,evidence) for key,text in questions.items()}
                    valid_credit = any(r['name']=='open_early' and r['valid'] for r in credits)
                    qa = {'drop_correct_abstention':answers['drop']['abstained'],
                          'remove_push_correct_abstention':answers['cf:remove_push']['abstained'],
                          'open_early_evidence_policy_correct':answers['cf:open_early']['abstained'] != valid_credit}
                    outputs[lang] = dict(explanation=explanation,questions=questions,answers=answers,qa=qa)
                # Truth boundary: only the scoring section sees simulator logs/success.
                for lang in outputs:
                    outputs[lang]['score'] = score(outputs[lang]['explanation'],rec,cfg.env.control_dt*cfg.slice.action_repeat)
                row = dict(seed=seed,disturbance_seed=disturbance_seed,condition=condition,mode=mode,
                           instruction=instruction,steps=rec.steps,belief=asdict(evidence),outputs=outputs,
                           truth=dict(success=rec.success,disturbances=rec.true_disturbances,events=rec.events))
                report['episodes'].append(row)
                save()
                print(condition,seed,mode,'success',rec.success,'cause',outputs['english']['explanation']['cause'],flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
