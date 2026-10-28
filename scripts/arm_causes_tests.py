"""Run the required regression cohort with a fresh Windows-compatible temp root."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.arm_causes import save


if __name__=='__main__':
    import pytest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basetemp',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=args.basetemp.resolve()
    if not base.is_relative_to(ROOT/'data') or base.exists():
        raise ValueError('Use a NEW basetemp inside this checkout/data; existing paths are never removed')
    original=os.mkdir
    def inherited(path,mode=0o777,*a,**kw):
        target=Path(path).resolve()
        if os.name=='nt' and mode==0o700 and (target==base or target.is_relative_to(base)):
            mode=0o777
        return original(path,mode,*a,**kw)
    os.mkdir=inherited
    class Results:
        def __init__(self):self.rows=[]
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed:
                self.rows.append({'nodeid':report.nodeid,'phase':report.when,'outcome':report.outcome,
                                  'duration':report.duration,'failure':str(report.longrepr) if report.failed else None})
    recorder=Results()
    paths=sorted(str(p) for pattern in ('test_causes*.py','test_language*.py') for p in Path('tests').glob(pattern))
    paths+=['tests/test_slice.py','tests/test_cognition.py']
    try:
        status=pytest.main(paths+['-q','-p','no:cacheprovider','--basetemp',str(base)],plugins=[recorder])
    finally:
        os.mkdir=original
    save(args.output,{'exit_code':int(status),'command':[sys.executable,*sys.argv],
                      'files':paths,'tests':recorder.rows,
                      'temporary_directory_note':'Only fresh children use inherited Windows permissions; existing ACLs are not changed.'})
    raise SystemExit(status)
