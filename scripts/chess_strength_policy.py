"""Train a small action prior from Stockfish's preferred teaching moves."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import chess
import numpy as np
import torch
from pai.chess.policy import DIM,LearnedPolicyPrior,action_features
from pai.chess.rules import LearnedChessRules
from pai.chess.representation import BoardAction
from pai.chess.teacher import observation,position_key
from pai.chess.hrm_value import mirror_observation
from pai.chess.distill import position_hash
from pai.chess.provenance import metadata

def read(path):
    with gzip.open(path,"rt") as stream: return [json.loads(line) for line in stream]

def prepare(rows,table):
    output = []
    for row in rows:
        board = chess.Board(row[0])
        moves = list(board.legal_moves)
        best = chess.Move.from_uci(row[2]) if row[2] else None
        if best not in moves: continue
        obs = observation(board)
        actions = [BoardAction.from_uci(m.uci()) for m in moves]
        output.append((action_features(obs,actions,table),moves.index(best),obs))
    return output

def padded(rows,mean,scale):
    size = max(len(row[0]) for row in rows)
    x = np.zeros((len(rows),size,DIM),dtype=np.float32)
    mask = np.zeros((len(rows),size),dtype=bool)
    for index,(features,_,_) in enumerate(rows):
        x[index,:len(features)] = (features-mean)/scale
        mask[index,:len(features)] = True
    return torch.tensor(x),torch.tensor(mask),torch.tensor([row[1] for row in rows])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=Path("runs/chess_strength_data"))
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--limit",type=int,default=12000)
    p.add_argument("--seed",type=int,default=42400)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    table = LearnedChessRules.load("runs/chess_rules/rules.json").attack_table
    train,val = read(args.directory/"train_mixed.jsonl.gz"),read(args.directory/"val.jsonl.gz")
    excluded = {position_hash(row[0]) for row in val}
    train = [row for row in train if position_hash(row[0]) not in excluded]
    rng = np.random.default_rng(args.seed)
    chosen = [train[i] for i in rng.choice(len(train),min(args.limit,len(train)),replace=False)]
    training,validation = prepare(chosen,table),prepare(val,table)
    all_x = np.concatenate([row[0] for row in training])
    mean,scale = all_x.mean(0),np.maximum(all_x.std(0),.1)
    xt,mt,yt = padded(training,mean,scale)
    xv,mv,yv = padded(validation,mean,scale)
    net = torch.nn.Sequential(torch.nn.Linear(DIM,64),torch.nn.ReLU(),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    opt = torch.optim.AdamW(net.parameters(),lr=.001)
    best,best_layers,history = -1,None,[]
    for epoch in range(args.epochs):
        order = torch.randperm(len(xt))
        net.train()
        for start in range(0,len(order),128):
            indices = order[start:start+128]
            logits = net(xt[indices]).squeeze(-1).masked_fill(~mt[indices],-1e9)
            loss = torch.nn.functional.cross_entropy(logits,yt[indices])
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            predicted = net(xv).squeeze(-1).masked_fill(~mv,-1e9).argmax(1)
            accuracy = float((predicted==yv).float().mean())
        history.append({"epoch":epoch+1,"validation_teacher_top1":accuracy})
        if accuracy > best:
            best = accuracy
            best_layers = [(module.weight.detach().numpy().T.copy(),module.bias.detach().numpy().copy()) for module in net if isinstance(module,torch.nn.Linear)]
        print(f"Policy epoch {epoch+1}: top1 {accuracy:.3f}",flush=True)
    keys = set()
    for _,_,obs in training+validation: keys.update((position_key(obs),position_key(mirror_observation(obs))))
    summary = {"training_positions":sorted(keys),"training_examples":len(training),"validation_examples":len(validation),
               "teacher_top1_accuracy":best,"seed":args.seed,"features":"generic relative token geometry and learned attack field"}
    LearnedPolicyPrior(best_layers,mean,scale,table,summary).save(args.directory/"policy.npz")
    report = {**metadata([args.seed,10400,10401,10402,10403,20400,20401,*range(36400,36408)],{"teacher":{"depth":8}}),
              "training":{k:v for k,v in summary.items() if k!="training_positions"},"history":history,
              "limitations":["teacher-generated policy labels; no engine at inference","development top1 accuracy is not playing strength"]}
    Path("results/chess_strength_policy_development.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__ == "__main__": main()
