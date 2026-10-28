"""Run only round-2 development collection and its mandatory pre-training gate."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    children = []
    for worker in range(2):
        log = Path(f'results/thinker_hier2_development_worker{worker}.log').open('a', encoding='utf8')
        command = [sys.executable, 'scripts/thinker_hier_round2.py', 'collect',
                   '--split', 'development', '--workers', '2', '--worker', str(worker)]
        children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    codes = []
    for process, log in children:
        codes.append(process.wait())
        log.close()
    if any(codes):
        raise SystemExit('Development collection incomplete; resume unchanged only')
    subprocess.run([sys.executable, 'scripts/thinker_hier_round2.py', 'gate'], check=True)
