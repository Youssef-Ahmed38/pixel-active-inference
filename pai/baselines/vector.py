"""Four CPU processes over anonymous stdio pipes (Windows named pipes denied)."""
import pickle,struct,subprocess,sys,re
from pathlib import Path
import numpy as np


def read_frame(stream):
    header=stream.read(4)
    if len(header)!=4:raise EOFError('Simulation worker closed; inspect its scoped log')
    size=struct.unpack('!I',header)[0];data=b''
    while len(data)<size:
        part=stream.read(size-len(data))
        if not part:raise EOFError('Incomplete simulation message')
        data+=part
    return pickle.loads(data)


def write_frame(stream,value):
    raw=pickle.dumps(value,protocol=5);stream.write(struct.pack('!I',len(raw))+raw);stream.flush()


def serve(index,seed_index):
    from pai.baselines.arm import ArmTask,CONDITIONS
    from pai.baselines.disturbances import disturbance_for
    task=ArmTask();rng=np.random.default_rng(120000+seed_index*100+index);counter=0
    condition=None;seed=None;disturbances=None;actions=[]
    def reset():
        nonlocal counter,condition,seed,disturbances,actions
        condition=str(rng.choice(CONDITIONS));seed=10000000+seed_index*1000000+index*100000+counter;counter+=1
        disturbances=disturbance_for(condition,rng);actions=[]
        return task.reset(seed,disturbances)
    try:
        while True:
            command,value=read_frame(sys.stdin.buffer)
            if command=='close':break
            if command=='reset':write_frame(sys.stdout.buffer,reset())
            elif command=='snapshot':
                write_frame(sys.stdout.buffer,dict(counter=counter,condition=condition,seed=seed,
                    disturbances=disturbances,actions=actions,rng=rng.bit_generator.state,vector=task.vector()))
            elif command=='restore':
                counter=value['counter'];condition=value['condition'];seed=value['seed']
                disturbances=value['disturbances'];actions=value['actions'];rng.bit_generator.state=value['rng']
                task.reset(seed,disturbances)
                for action in actions:
                    _,_,done,_=task.step(action)
                    if done:raise ValueError('Checkpoint contains a completed current episode')
                if not np.array_equal(task.vector(),value['vector']):raise ValueError('Environment replay is not bit-exact')
                write_frame(sys.stdout.buffer,task.vector())
            elif command=='step':
                actions.append(np.asarray(value).copy())
                obs,reward,done,truncated=task.step(value);following=obs;summary=None
                if done:
                    score=task.score();summary=dict(seed=seed,condition=condition,success=bool(score['success']),steps=score['steps'],return_=score['return_']);following=reset()
                write_frame(sys.stdout.buffer,(obs,reward,done,truncated,following,summary))
    finally:task.close()


class Vector:
    def __init__(self,n,seed_index,tag='probe'):
        self.processes=[];self.logs=[]
        for i in range(n):
            path=Path(f'results/arm_baselines_worker_{tag}_{seed_index}_{i}.log');handle=path.open('a',encoding='utf8')
            p=subprocess.Popen([sys.executable,'-u','-m','pai.baselines.vector',str(i),str(seed_index)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=handle)
            self.processes.append(p);self.logs.append((path,handle))
    def reset(self):
        for p in self.processes:write_frame(p.stdin,('reset',None))
        return np.stack([read_frame(p.stdout) for p in self.processes])
    def step(self,actions):
        for p,a in zip(self.processes,actions):write_frame(p.stdin,('step',a))
        rows=[read_frame(p.stdout) for p in self.processes]
        return [np.stack([r[i] for r in rows]) for i in range(5)],[r[5] for r in rows]
    def snapshot(self):
        for p in self.processes:write_frame(p.stdin,('snapshot',None))
        return [read_frame(p.stdout) for p in self.processes]
    def restore(self,states):
        if len(states)!=len(self.processes):raise ValueError('Worker count changed')
        for p,state in zip(self.processes,states):write_frame(p.stdin,('restore',state))
        return np.stack([read_frame(p.stdout) for p in self.processes])
    def close(self):
        for p in self.processes:
            try:write_frame(p.stdin,('close',None))
            except (BrokenPipeError,OSError):pass
        for p in self.processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.terminate();p.wait()
            p.stdin.close();p.stdout.close()
        for path,handle in self.logs:
            handle.close();text=path.read_text(encoding='utf8');path.write_text(re.sub(r'File "[^\n"]+"','File "<worker-source>"',text),encoding='utf8')


if __name__=='__main__':serve(int(sys.argv[1]),int(sys.argv[2]))
