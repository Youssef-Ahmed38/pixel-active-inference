"""Opt-in ordinary-pipe UCI transport for restricted Windows evaluation hosts.

Teacher/opponent utility only; never imported by the learned player. Preserves
history, UCI limits and multipv score frames. No engine-derived adjudication.
"""
from pathlib import Path
from queue import Queue,Empty
from threading import Thread
from types import SimpleNamespace
import subprocess
import chess
import chess.engine


class SyncUCI:
    def __init__(self,command,timeout=30.,**unused):
        if isinstance(command,(list,tuple)):command=list(command)
        else:command=[command]
        command[0]=str(Path(command[0]).resolve())
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,text=True,bufsize=1,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.lines=Queue();self.timeout=timeout;self.id={};self.options=set()
        def reader():
            for line in self.process.stdout:self.lines.put(line.strip())
            self.lines.put(None)
        Thread(target=reader,daemon=True).start()
        self.send('uci')
        for line in self.until('uciok'):
            if line.startswith('id '):
                _,key,value=line.split(' ',2);self.id[key]=value
            if line.startswith('option name '):self.options.add(line[12:].split(' type ')[0])

    @classmethod
    def popen_uci(cls,command,**kwargs):return cls(command,**kwargs)

    def send(self,text):
        self.process.stdin.write(text+'\n');self.process.stdin.flush()

    def until(self,prefix,timeout=None):
        while True:
            try:line=self.lines.get(timeout=timeout or self.timeout)
            except Empty:raise TimeoutError('UCI response timed out') from None
            if line is None:raise RuntimeError('UCI process exited before response')
            yield line
            if line.startswith(prefix):break

    def ready(self):
        self.send('isready');list(self.until('readyok'))

    def configure(self,options):
        for name,value in options.items():
            if name not in self.options:raise ValueError('unsupported UCI option: '+name)
            if isinstance(value,bool):value=str(value).lower()
            self.send('setoption name '+name+(' value '+str(value) if value is not None else ''))
        self.ready()

    def _search(self,board,limit,multipv):
        self.configure({'MultiPV':multipv})
        position='position fen '+board.root().fen()
        if board.move_stack:position+=' moves '+' '.join(m.uci() for m in board.move_stack)
        self.send(position)
        go='go'
        for name,value in [('movetime',None if limit.time is None else max(1,int(limit.time*1000))),
                           ('depth',limit.depth),('nodes',limit.nodes)]:
            if value is not None:go+=f' {name} {value}'
        if go=='go':raise ValueError('bounded UCI search required')
        self.send(go);infos={};best=None
        for line in self.until('bestmove',timeout=max(self.timeout,(limit.time or 0)+10)):
            words=line.split()
            if line.startswith('bestmove'):
                if len(words)>1 and words[1] not in ('(none)','0000'):best=chess.Move.from_uci(words[1])
            elif line.startswith('info ') and 'score' in words and 'pv' in words:
                j=words.index('score');kind,value=words[j+1:j+3]
                if kind not in ('cp','mate'):continue
                score=chess.engine.Cp(int(value)) if kind=='cp' else chess.engine.Mate(int(value))
                pv=[chess.Move.from_uci(m) for m in words[words.index('pv')+1:]]
                rank=int(words[words.index('multipv')+1]) if 'multipv' in words else 1
                infos[rank]={'score':chess.engine.PovScore(score,board.turn),'pv':pv}
        return best,[infos[k] for k in sorted(infos)]

    def analyse(self,board,limit,multipv=None,**unused):
        _,infos=self._search(board,limit,multipv or 1)
        if not infos:raise RuntimeError('UCI search returned no scored PV')
        return infos if multipv is not None else infos[0]

    def play(self,board,limit,**unused):
        move,_=self._search(board,limit,1);return SimpleNamespace(move=move)

    def quit(self):
        if self.process.poll() is None:
            self.send('quit')
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdin.close();self.process.stdout.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.quit()


def install_sync_engine():
    """Explicit script/process opt-in; ordinary package imports change nothing."""
    chess.engine.SimpleEngine.popen_uci=SyncUCI.popen_uci
