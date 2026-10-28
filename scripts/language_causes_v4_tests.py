"""Run the language gates with inherited Windows temp-directory permissions."""
import argparse
import os
from pathlib import Path
import sys
import pytest
from language_causes_v4_train import save

class Results:
    def __init__(self):
        self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:
            self.rows.append(dict(test=report.nodeid,phase=report.when,outcome=report.outcome))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basetemp',default='data/pytest_language_v4_inherited_20261028')
    ap.add_argument('--output',default='results/language_causes_v4_tests.json')
    args=ap.parse_args()
    base=Path(args.basetemp).resolve()
    if base.exists():
        raise FileExistsError('Use a fresh basetemp; existing directories are never removed')
    root=(Path.cwd()/'data').resolve()
    if root not in base.parents:
        raise ValueError('Temporary directory must be under project data/')
    original=os.mkdir
    def inherited(path,mode=0o777,*,dir_fd=None):
        resolved=Path(path).resolve()
        if resolved==base or base in resolved.parents:
            mode=0o777
        return original(path,mode,dir_fd=dir_fd)
    results=Results()
    os.mkdir=inherited
    try:
        code=pytest.main([*[str(p) for p in Path('tests').glob('test_language*.py')],
                          'tests/test_slice.py','tests/test_cognition.py','-q','-p','no:cacheprovider',
                          '--basetemp='+args.basetemp],plugins=[results])
    finally:
        os.mkdir=original
    save(args.output,dict(exit_code=int(code),tests=results.rows,command=['python',*sys.argv],
                          note='Only new test directories use inherited permissions; no production code patched'))
    raise SystemExit(code)
