"""Execute the registered round-3 gates and fixed comparison without final tuning."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3 import PREFIX, register


def sanitize(path):
    text = path.read_text(encoding='utf8')
    def clean(match):
        source = match.group(1).replace('\\', '/')
        root = Path.cwd().as_posix()+'/'
        return 'File "'+(source[len(root):] if source.startswith(root) else '<python-library>')+'"'
    path.write_text(re.sub(r'File "([^"]+)"', clean, text), encoding='utf8')


def call(name, args):
    log = Path(PREFIX+'stage_'+name+'.log')
    with log.open('a', encoding='utf8') as handle:
        result = subprocess.run([sys.executable, *args], stdout=handle, stderr=subprocess.STDOUT)
    sanitize(log)
    save(PREFIX+'stage_'+name+'.json', dict(command=['python', *args], exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError('Stage failed: '+name+'; retain failure and resume unchanged only')
    print('Completed', name, flush=True)


def collect(split, workers=2):
    children = []
    for worker in range(workers):
        manifest = Path(PREFIX+split+f'_manifest{worker}.json')
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf8'))['complete']:
            continue
        path = Path(PREFIX+split+f'_worker{worker}.log')
        handle = path.open('a', encoding='utf8')
        args = ['scripts/thinker_hier_round3_collect.py', 'collect', '--split', split,
                '--workers', str(workers), '--worker', str(worker)]
        children.append((subprocess.Popen([sys.executable, *args], stdout=handle,
                         stderr=subprocess.STDOUT), handle, path, args))
    # CUDA can hang in renderer teardown after a fatal allocation error. Terminate
    # only that own failed subprocess tree, never another training job.
    stopped = []
    while any(p.poll() is None for p, _, _, _ in children):
        for process, handle, path, args in children:
            if process.poll() is None and 'CUDA error: out of memory' in path.read_text(encoding='utf8'):
                import psutil
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.terminate()
                process.terminate(); stopped.append(args)
        time.sleep(5)
    failed = []
    for process, handle, path, args in children:
        handle.close(); sanitize(path)
        if process.returncode:
            failed.append((path, args))
    if failed:
        if not all('CUDA error: out of memory' in path.read_text(encoding='utf8') for path, _ in failed):
            raise RuntimeError('Collection failure; inspect saved logs: '+split)
        save(PREFIX+split+'_oom_recovery.json', dict(failures=[args for _, args in failed],
            stopped_failed_subprocesses=stopped, policy='Resume missing keys serially, same settings/seeds; completed episodes unchanged'))
        for path, args in failed:
            with path.open('a', encoding='utf8') as handle:
                result = subprocess.run([sys.executable, *args], stdout=handle, stderr=subprocess.STDOUT)
            sanitize(path)
            if result.returncode:
                raise RuntimeError('Serial recovery failed; retain failure: '+split)
    print('Collection complete:', split, flush=True)


def pipeline():
    plan = register()
    a = json.loads(Path(PREFIX+'gate_a.json').read_text(encoding='utf8'))
    save(PREFIX+'gate_a_status.json', dict(passed=a['passed'], gate_sha256=digest(PREFIX+'gate_a.json')))
    call('report_gate_a', ['scripts/thinker_hier_round3_report.py', 'gates'])
    if not a['passed']:
        print('Gate A failed; stopped before collection/training', flush=True); return
    collect('calibration')
    if not Path(PREFIX+'reference_calibration.json').exists():
        call('calibrate', ['scripts/thinker_hier_round3_collect.py', 'calibrate'])
    collect('development')
    if not Path(PREFIX+'gate_b.json').exists():
        call('gate_b', ['scripts/thinker_hier_round3_collect.py', 'gate-b'])
    call('report_gate_b', ['scripts/thinker_hier_round3_report.py', 'gates'])
    if not json.loads(Path(PREFIX+'gate_b.json').read_text(encoding='utf8'))['passed']:
        print('Gate B failed; stopped before training/final collection', flush=True); return
    if json.loads(Path(PREFIX+'tests_ready.json').read_text(encoding='utf8'))['exit_code']:
        raise ValueError('Required tests must pass before training')
    collect('train')
    for seed in plan['training_seeds']:
        for kind in plan['models']:
            prefix = PREFIX+f'{kind}_s{seed}'
            if Path(prefix+'_training.json').exists():
                continue
            call(f'train_{kind}_{seed}', ['scripts/thinker_hier_round3_train.py', '--kind', kind, '--seed', str(seed)])
    call('development_readouts', ['scripts/thinker_hier_round3_evaluate.py', 'development'])
    if not Path(PREFIX+'freeze.json').exists():
        call('freeze', ['scripts/thinker_hier_round3_evaluate.py', 'freeze'])
    collect('final')
    call('final_scores', ['scripts/thinker_hier_round3_evaluate.py', 'final'])
    if not Path(PREFIX+'references_final.json').exists():
        call('references_final', ['scripts/thinker_hier_round3_references.py', 'final'])
    call('report_final', ['scripts/thinker_hier_round3_report.py', 'final'])
    call('audit', ['scripts/thinker_hier_round3_audit.py'])


if __name__ == '__main__':
    pipeline()
