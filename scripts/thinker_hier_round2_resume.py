"""Resume only the failed development shard, after the surviving shard exits."""
import json
import subprocess
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    surviving = Path('results/thinker_hier2_development_manifest0.json')
    while not json.loads(surviving.read_text(encoding='utf8'))['complete']:
        time.sleep(5)
    with Path('results/thinker_hier2_development_worker1.log').open('a', encoding='utf8') as log:
        log.write('\nResuming missing keys only, with unchanged configuration and seeds.\n')
        log.flush()
        subprocess.run([sys.executable, 'scripts/thinker_hier_round2.py', 'collect',
            '--split', 'development', '--workers', '2', '--worker', '1'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, 'scripts/thinker_hier_round2.py', 'gate'], check=True)
