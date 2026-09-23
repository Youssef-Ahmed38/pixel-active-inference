"""The thinking layer: an HRM reasoning network trained to imitate an exact Bayesian teacher."""

from pai.thinker.baselines import FixedDepthTransformer, MLPThinker
from pai.thinker.hrm import HRM, SegmentOut, ThinkResult
from pai.thinker.multicause import CAUSE_LIB, MultiCauseConfig, MultiCauseTask
from pai.thinker.synthetic import CAUSES, CauseTask, TaskConfig

__all__ = ["HRM", "SegmentOut", "ThinkResult", "FixedDepthTransformer", "MLPThinker", "CAUSES", "CauseTask", "TaskConfig",
           "CAUSE_LIB", "MultiCauseConfig", "MultiCauseTask"]
