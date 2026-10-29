"""Launch the registered runner with independent handles and no parent job lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys

from arm_baseline_plan import PREFIX,save


if __name__=='__main__':
    import psutil
    for process in psutil.process_iter():
        try:
            if any(a in ('scripts/arm_baseline_run.py','scripts/arm_baseline_train.py') for a in process.cmdline()):
                raise RuntimeError('Registered runner/trainer already active; do not duplicate')
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    env=dict(os.environ,PYTHONPATH='.;scripts',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    flags=0
    if os.name=='nt':
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    command=[sys.executable,'-u','scripts/arm_baseline_run.py']
    with open(PREFIX+'detached_stdout.log','ab') as out,open(PREFIX+'detached_stderr.log','ab') as err:
        process=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,close_fds=True,creationflags=flags,
            **({'start_new_session':True} if os.name!='nt' else {}))
    save(PREFIX+'detached_launch.json',dict(pid=process.pid,command=['python','-u','scripts/arm_baseline_run.py'],
        launcher=['python','scripts/arm_baseline_launch.py'],independent_stdin_stdout_stderr=True,
        detached=True,breakaway_from_parent_job=os.name=='nt'))
    print(json.dumps({'detached_pid':process.pid}))
