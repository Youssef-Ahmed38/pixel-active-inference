"""Execute a study entry point with Windows-safe result replacement only."""
from pathlib import Path
import runpy,sys
from pai.thinker.hier_io import install_replace_retry
if __name__=='__main__':
    script=sys.argv[1];sys.argv=sys.argv[1:]
    original=install_replace_retry()
    try:runpy.run_path(script,run_name='__main__')
    finally:Path.replace=original
