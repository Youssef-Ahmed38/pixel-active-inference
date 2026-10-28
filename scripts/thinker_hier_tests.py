"""Run thinker/slice/cognition gates with fresh inherited-permission test temp directories."""
import argparse,os,sys
from pathlib import Path
import pytest
import torch
torch.set_num_threads(1)
from thinker_hier_collect import save
class Results:
    def __init__(self):self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.rows.append(dict(test=report.nodeid,outcome=report.outcome,phase=report.when))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--basetemp',default='data/pytest_thinker_hier_20261028');ap.add_argument('--output',default='results/thinker_hier_tests.json');args=ap.parse_args()
    base=Path(args.basetemp).resolve();root=(Path.cwd()/'data').resolve()
    if base.exists() or root not in base.parents:raise ValueError('Use a new temp directory under data/')
    old=os.mkdir
    def inherited(p,mode=0o777,*,dir_fd=None):
        resolved=Path(p).resolve()
        return old(p,0o777 if resolved==base or base in resolved.parents else mode,dir_fd=dir_fd)
    os.mkdir=inherited;results=Results()
    try:code=pytest.main([*[str(p) for p in Path('tests').glob('test_thinker*.py')],'tests/test_slice.py','tests/test_cognition.py','-q','-p','no:cacheprovider','--basetemp='+args.basetemp],plugins=[results])
    finally:os.mkdir=old
    save(args.output,dict(exit_code=int(code),tests=results.rows,command=['python',*sys.argv]))
    raise SystemExit(code)
