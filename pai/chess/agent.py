"""Pure learned chess player: public beliefs in, commands out, no oracle imports."""
from pathlib import Path
from pai.chess.state import GameState

class AgentPlayer:
    def __init__(self,config):
        from threadpoolctl import threadpool_limits
        self.thread_limit = threadpool_limits(limits=1)
        import torch
        torch.set_num_threads(1)
        from pai.chess import LearnedChessRules,LearnedChessStateModel,LearnedChessEvaluator,LearnedChessSearch
        from pai.chess.hrm_value import load_value
        directory = Path(config.get("directory","runs/chess_rules"))
        rules = LearnedChessRules.load(directory/"rules.json")
        state = LearnedChessStateModel.load(directory/"state.json",rules)
        endings = LearnedChessEvaluator.load(directory/"endings.json",state)
        value = load_value(config.get("value",directory/"value.json"))
        ordering = load_value(config["ordering"]) if config.get("ordering") else None
        prior = None
        if config.get("policy"):
            from pai.chess.policy import LearnedPolicyPrior
            prior = LearnedPolicyPrior.load(config["policy"])
        self.planner = LearnedChessSearch(state,endings,value,ordering,prior)
        self.config = dict(config)
        self.warmed = False

    def prepare(self,belief):
        if not self.warmed:
            self.planner.search(GameState.from_belief(belief),max_depth=1,max_nodes=32)
            self.warmed = True

    def command_from_belief(self,belief):
        state = GameState.from_belief(belief)
        c = self.config
        result = self.planner.search(state,max_depth=c.get("max_depth",32),max_nodes=c.get("nodes",10**9),
                                     time_limit=c.get("time"),max_quiescence=c.get("quiescence",0),reuse_pv=c.get("reuse_pv",False))
        return result.command(),{"depth":result.completed_depth,"nodes":result.nodes,
                                 "seconds":result.elapsed_seconds,"budget_exhausted":result.budget_exhausted}

    def close(self): pass
