"""Run the scoped tests with new inherited-permission temporary directories."""
import os
from pathlib import Path
import pytest
import torch
from arm_baseline_plan import save,PREFIX

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--basetemp',default='data/pytest_arm_baselines_20261028');ap.add_argument('--output',default=PREFIX+'tests.json');args=ap.parse_args()
    torch.set_num_threads(1)
    base=Path(args.basetemp).resolve()
    if (Path.cwd()/'data').resolve() not in base.parents:raise ValueError('Temporary directory must be under data/')
    if base.exists():raise ValueError('Choose a fresh temporary directory')
    old=os.mkdir
    def inherited(p,mode=0o777,*,dir_fd=None):
        resolved=Path(p).resolve()
        return old(p,0o777 if resolved==base or base in resolved.parents else mode,dir_fd=dir_fd)
    os.mkdir=inherited;rows=[]
    class Results:
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed:rows.append(dict(test=report.nodeid,outcome=report.outcome))
    try:code=pytest.main(['tests/test_arm_baselines.py','tests/test_slice.py','tests/test_cognition.py','-q','-p','no:cacheprovider','--basetemp='+str(base)],plugins=[Results()])
    finally:os.mkdir=old
    save(args.output,dict(exit_code=int(code),tests=rows,command=['python','scripts/arm_baseline_tests.py','--basetemp',args.basetemp,'--output',args.output]))
    raise SystemExit(code)
