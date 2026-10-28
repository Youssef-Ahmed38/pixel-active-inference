"""Register and execute the arm's frozen final study; never fit on final data."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.arm_causes import load,save,sources,sha,inventory


def commands(tag,spec):
    prefix=f'results/arm_causes_{tag}'
    common=['--regime',spec['regime'],'--calibration',spec['calibration'],'--model',spec['model']]
    result=[]
    for controller,count,suffix in [('baseline',spec['count'],'final'),('learned',spec['controller_count'],'controller')]:
        data=prefix+f'_{suffix}_data.json.gz';score=prefix+f'_{suffix}_scores.json.gz'
        audit=prefix+('_audit.json.gz' if suffix=='final' else '_controller_audit.json.gz')
        result += [ ['scripts/arm_causes.py','collect',*common,'--controller',controller,'--seed',str(spec['seed']),
                     '--count',str(count),'--output',data],
                    ['scripts/arm_causes.py','score','--input',data,'--model',spec['model'],'--output',score],
                    ['scripts/arm_causes_audit.py','audit','--input',data,'--model',spec['model'],'--scores',score,'--output',audit]]
    result += [ ['scripts/arm_causes_audit.py','online','--input',prefix+'_final_data.json.gz',
                 '--model',spec['model'],'--output',prefix+'_online.json.gz'],
                ['scripts/arm_causes_audit.py','control','--input',prefix+'_final_data.json.gz',
                 '--other',prefix+'_controller_data.json.gz','--output',prefix+'_control.json.gz']]
    return result


def register(path):
    seen=inventory();used={s for values in seen.values() if isinstance(values,list) for s in values}
    regimes={}
    for tag,regime,seed in [('exact','exact',850000),('pixel','pixel_slots',860000)]:
        model=f'results/arm_causes_{tag}_model_frozen.json.gz'
        calibration=f'results/arm_causes_{tag}_calibration.json.gz'
        frozen=load(model)
        assert frozen['frozen_sources']==sources() and frozen['regime']==regime
        assert frozen['calibration_sha256']==sha(calibration)
        assert not set(range(seed,seed+20))&used,'Final seeds already recorded'
        regimes[tag]={'regime':regime,'seed':seed,'count':20,'controller_count':5,
                      'model':model,'model_sha256':sha(model),'calibration':calibration,
                      'calibration_sha256':sha(calibration)}
    plan={'kind':'preregistered_arm_cause_gate','regimes':regimes,'frozen_sources':sources(),
          'seed_inventory_before_final':seen,
          'commands':[cmd for tag,spec in regimes.items() for cmd in commands(tag,spec)],
          'study_script_sha256':sha(__file__),
          'rules':['No further fitting or selection after final collection starts',
                   'Retain all final episodes and failures; no seed replacement',
                   '20 layouts / five conditions per regime; exploratory controller subset first five layouts',
                   'Same-trace diagnostic pairing; controller pairing by initial layout and scheduled condition']}
    save(path,plan)
    print('Registered final seeds: exact 850000–850019; camera 860000–860019; controller first five layouts each.',flush=True)


def run(path):
    plan=load(path)
    assert plan['study_script_sha256']==sha(__file__)
    for spec in plan['regimes'].values():
        assert sha(spec['model'])==spec['model_sha256']
        assert sha(spec['calibration'])==spec['calibration_sha256']
    for command in plan['commands']:
        assert sources()==plan['frozen_sources'],'Frozen inference source changed'
        output=Path(command[command.index('--output')+1])
        if output.exists():
            old=load(output)
            if command[1]=='collect' and not old['complete']:
                command=[*command,'--resume']
            else:
                if command[1]=='collect':assert not old['source_changes_during_run']
                print('Already recorded; retaining',output,flush=True)
                continue
        print('Running:',subprocess.list2cmdline([sys.executable,*command]),flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','run'))
    parser.add_argument('--plan',type=Path,default=Path('results/arm_causes_plan.json'))
    args=parser.parse_args()
    {'register':register,'run':run}[args.mode](args.plan)
