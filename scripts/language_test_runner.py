"""Run unchanged regression tests with inheriting Windows temporary-directory ACLs.

Python's Windows mkdir(mode=0o700) can exclude the managed executor's restricted
token. Only fresh pytest temp directories within the requested base use normal
inherited permissions; no existing ACL or production file permissions are changed.
"""
import argparse
import os
from pathlib import Path
import sys
import pytest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basetemp',required=True,type=Path)
    args=ap.parse_args()
    base=args.basetemp.resolve()
    if base.exists():
        raise FileExistsError('Choose a fresh temporary directory: '+str(base))
    workspace=Path.cwd().resolve()
    if not base.is_relative_to(workspace):
        raise ValueError('Test temporary directory must stay in workspace')
    original=os.mkdir
    def inherited_mkdir(path,mode=0o777,*a,**kw):
        target=Path(path).resolve()
        if os.name=='nt' and mode==0o700 and (target==base or target.is_relative_to(base)):
            mode=0o777
        return original(path,mode,*a,**kw)
    os.mkdir=inherited_mkdir
    tests=[str(p) for p in Path('tests').glob('test_language*.py')]
    tests+=['tests/test_slice.py','tests/test_cognition.py']
    sys.exit(pytest.main([*tests,'-q','-p','no:cacheprovider','--basetemp',str(base)]))
