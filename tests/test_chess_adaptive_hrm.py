"""Adaptive recurrence, symmetry, persistence, and the no-oracle move boundary."""
import builtins
from pathlib import Path
import numpy as np
import pytest
import torch
from pai.chess.adaptive_hrm import AdaptiveHRMValue,stopping_targets
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_value import load_value,mirror_observation
from pai.chess.representation import BoardObservation
from pai.thinker.hrm import HRM


def tiny():
    torch.manual_seed(13)
    cfg=dict(in_dim=16,ctx_dim=5,max_len=64,heads={'value':1},d_model=16,n_heads=2,
             h_layers=1,l_layers=1,h_cycles=1,l_steps=1,max_segments=4,patch=8)
    return AdaptiveHRMValue(HRM(**cfg),LearnedPositionValue(np.arange(391)/100),cfg)


def board():
    pieces=[0]*64;pieces[4]=6;pieces[60]=12;pieces[12]=1
    return BoardObservation(tuple(pieces),1,(0,0,0,0),64)


def test_teacher_benefit_targets_and_compute_cost():
    errors=torch.tensor([[1.,100.,100.],[1.,90.,100.],[1.,80.,99.],[1.,70.,98.]],requires_grad=True)
    labels=stopping_targets(errors,2.)
    assert labels[:,0].tolist()==[0,0,0,0]
    assert labels[:,1].tolist()==[1,1,1,0]
    assert labels[:,2].tolist()==[0,0,0,0]
    assert not labels.requires_grad


def test_halt_head_changes_actual_compute_and_forced_cap():
    v=tiny();torch.nn.init.zeros_(v.model.q_head.weight)
    with torch.no_grad():v.model.q_head.bias.copy_(torch.tensor([1.,-1.]))
    v.predict(board());assert v.halting_stats()['segments_histogram']=={1:2}
    v.reset_diagnostics();v.predict_many([board()],segments=4)
    assert v.halting_stats()['segments_histogram']=={4:2}
    with pytest.raises(ValueError):v.predict_many([board()],segments=0)


def test_adaptive_symmetry_batch_io_and_empty(tmp_path):
    v=tiny();o=board();pred=v.predict_many([o,mirror_observation(o)])
    assert pred[0]==pytest.approx(-pred[1],abs=1e-3)
    assert v.predict(o)==pytest.approx(pred[0],abs=1e-3)
    path=tmp_path/'adaptive.pt';v.save(path);restored=load_value(path)
    assert isinstance(restored,AdaptiveHRMValue)
    assert restored.predict(o)==pytest.approx(pred[0],abs=1e-3)
    assert restored.predict_many([]).shape==(0,)


def test_adaptive_inference_has_no_teacher_import(monkeypatch):
    v=tiny();original=builtins.__import__
    def reject(name,*a,**kw):
        if name=='chess' or name.startswith('chess.') or name=='pai.chess.teacher':raise AssertionError('oracle')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',reject)
    assert np.isfinite(v.predict(board()))


@pytest.mark.skipif(not Path('results/chess_hrm_adaptive_model.pt').exists() or
                    not Path('runs/chess_rules/rules.json').exists(),reason='optional learned checkpoints absent')
def test_real_search_calls_adaptive_value_without_engine(monkeypatch):
    import chess,chess.engine
    from pai.chess.agent import AgentPlayer
    from pai.chess.teacher import ChessTeacher
    player=AgentPlayer({'value':'results/chess_hrm_adaptive_model.pt',
                        'policy':'runs/chess_strength_data/policy.npz','time':.1,'quiescence':2})
    belief=ChessTeacher().observe();player.prepare(belief)
    original=builtins.__import__
    def reject(name,*a,**kw):
        if name=='chess' or name.startswith('chess.') or name=='pai.chess.teacher':raise AssertionError('oracle')
        return original(name,*a,**kw)
    def blocked(*a,**kw):raise AssertionError('engine in agent path')
    monkeypatch.setattr(builtins,'__import__',reject)
    monkeypatch.setattr(chess.engine.SimpleEngine,'popen_uci',blocked)
    command,stats=player.command_from_belief(belief)
    assert command.name=='move_piece' and stats['halting']['boards']>0


def test_sync_uci_limits_history_and_score_frame_without_process():
    import chess,chess.engine
    from pai.chess.uci_sync import SyncUCI
    transport=object.__new__(SyncUCI);sent=[]
    transport.timeout=30
    transport.configure=lambda options:sent.append(options)
    transport.send=sent.append
    transport.until=lambda *a,**kw:iter(['info depth 8 multipv 1 score cp 32 pv e7e5',
                                       'info depth 8 multipv 2 score cp 20 pv c7c5','bestmove e7e5'])
    b=chess.Board();b.push_uci('e2e4')
    move,infos=transport._search(b,chess.engine.Limit(time=.2),2)
    assert move.uci()=='e7e5' and len(infos)==2
    assert infos[0]['score'].white().score()==-32
    assert sent[1].endswith(' moves e2e4') and sent[2]=='go movetime 200'
