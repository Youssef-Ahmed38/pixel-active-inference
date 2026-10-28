"""Bounded retries for Windows result-file sharing violations; no data transformation."""
from pathlib import Path
import time


def install_replace_retry():
    original=Path.replace
    def replace(source,target):
        a,b=Path(source).resolve(),Path(target).resolve();root=(Path.cwd()/'results').resolve()
        ours=root in a.parents and root in b.parents and a.name.startswith('thinker_hier_') and b.name.startswith('thinker_hier_')
        if not ours:return original(source,target)
        for i in range(100):
            try:return original(source,target)
            except PermissionError:
                if i==99:raise
                time.sleep(.05)
    Path.replace=replace
    return original
