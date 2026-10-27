"""Teacher-learned action ordering over public tokens and generic geometry."""
import json
from pathlib import Path
import numpy as np
from pai.chess.hrm_value import mirror_observation
from pai.chess.rules import spatial_features

DIM = 133

def action_features(obs, actions, table):
    a = np.asarray([(m.source,m.destination,m.promotion) for m in actions],dtype=np.int16).reshape(-1,3)
    if not obs.turn:
        obs = mirror_observation(obs)
        a = a.copy()
        a[:,:2] ^= 56
    pieces = np.asarray(obs.pieces)
    base = spatial_features(obs,a,table)
    identities = np.concatenate((np.eye(13)[pieces[a[:,0]]],np.eye(13)[pieces[a[:,1]]]),axis=1)
    centers,counts = [],[]
    for code in range(1,13):
        cells = np.flatnonzero(pieces == code)
        centers.append([np.mean(cells % 8),np.mean(cells // 8)] if len(cells) else [0,0])
        counts.append(len(cells))
    centers = np.asarray(centers)
    relative = []
    for index in (0,1):
        coordinates = np.column_stack((a[:,index] % 8,a[:,index] // 8))
        relative.append((coordinates[:,None,:]-centers[None]).reshape(-1,24))
    return np.concatenate((base,identities,*relative,np.tile(counts,(len(a),1))),axis=1).astype(np.float32)

class LearnedPolicyPrior:
    def __init__(self,layers,mean,scale,table,summary=None):
        self.layers = [(np.asarray(w,dtype=np.float32),np.asarray(b,dtype=np.float32)) for w,b in layers]
        self.mean,self.scale = np.asarray(mean,dtype=np.float32),np.asarray(scale,dtype=np.float32)
        self.table = np.asarray(table,dtype=np.int8)
        self.summary = summary or {}
        if self.mean.shape != (DIM,) or self.scale.shape != (DIM,) or np.any(self.scale <= 0):
            raise ValueError("matching positive policy feature scales required")
        if not self.layers or self.layers[0][0].shape[0] != DIM or self.layers[-1][0].shape[1] != 1:
            raise ValueError("unexpected policy layers")
        if not all(np.all(np.isfinite(x)) for x in (self.mean,self.scale,*[v for pair in self.layers for v in pair])):
            raise ValueError("finite policy parameters required")

    def predict_actions(self,obs,actions):
        if not actions: return np.empty(0)
        h = (action_features(obs,actions,self.table)-self.mean)/self.scale
        for index,(w,b) in enumerate(self.layers):
            h = h @ w + b
            if index < len(self.layers)-1: h = np.maximum(h,0)
        return h[:,0]

    def save(self,path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        data = {"mean":self.mean,"scale":self.scale,"table":self.table,
                "summary":np.frombuffer(json.dumps(self.summary).encode(),dtype=np.uint8)}
        for index,(w,b) in enumerate(self.layers): data[f"w{index}"],data[f"b{index}"] = w,b
        with path.open("wb") as stream: np.savez_compressed(stream,**data)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            layers = [(data[f"w{i}"],data[f"b{i}"]) for i in range(sum(k.startswith("w") for k in data.files))]
            return cls(layers,data["mean"],data["scale"],data["table"],json.loads(bytes(data["summary"]).decode()))
