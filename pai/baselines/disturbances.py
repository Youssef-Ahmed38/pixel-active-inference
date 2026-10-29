"""Execute the unchanged distribution function without importing evaluation/torch.

Only the simulator worker uses this sampler; its outputs never enter policy
observations or rewards. AST extraction avoids duplicating the authored settings.
"""
import ast
from pathlib import Path
import numpy as np

tree=ast.parse(Path('scripts/slice_eval.py').read_text(encoding='utf8'))
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='disturbance_for')
namespace={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),'scripts/slice_eval.py','exec'),namespace)
disturbance_for=namespace['disturbance_for']
