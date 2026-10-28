"""Fresh maze/fixture bridge check for the frozen learned chain (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pai.discovery.key_chain_body import SituatedChainBody
from pai.discovery.key_chain_recipe import RecipeDiscoveryAgent, FailureLikelihood
from pai.maze.env import MazeWorld
from pai.maze.key_chain import run_key_chain
from scripts.key_chain_v2_eval import (scene, restore_memory, record_log, explain_record,
                                       summarize, source_hashes, copy_record)


def evaluate(args):
    frozen = json.loads(args.model.read_text(encoding='utf-8'))
    if frozen['sources'] != source_hashes():
        raise ValueError('Frozen inference sources changed')
    seeds = set(range(args.seed, args.seed + args.layouts * 2))
    used = set(frozen['development_seeds'])
    for p in (ROOT / 'results').glob('key_chain_v2_*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        used.update(r['seed'] for r in d.get('episodes', []))
        if 'episode' in d:
            used.add(d['episode']['seed'])
    used.update(s for v in frozen['seed_inventory'].values() if isinstance(v, list) for s in v)
    if seeds & used:
        raise ValueError('Previously used seeds')
    source_paths = ['pai/discovery/key_chain_body.py', 'scripts/key_chain_v2_situated.py']
    sources = {**source_hashes(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}}
    likelihood = FailureLikelihood.from_dict(frozen['failure_model'])
    rows = []
    for i in range(args.layouts):
        name = 'hd_lever_pull_right' if i < args.layouts // 2 else 'hd_knob_pull_right'
        for j, case in enumerate(('key', 'both')):
            seed = args.seed + 2 * i + j
            modes = {}
            for mode in ('cold', 'warm'):
                fixture = scene(seed, name, case)
                maze = MazeWorld(seed=seed, size=args.size)
                try:
                    body = SituatedChainBody(maze, fixture, seed=seed, move_budget=args.moves)
                except ValueError as exc:
                    modes[mode] = {'success': False, 'actions': 0, 'actions_to_success': None,
                                   'moves': 0, 'case': case, 'seed': seed, 'type': name,
                                   'invalid_layout': str(exc)}
                    continue
                agent = RecipeDiscoveryAgent(seed=seed, memory=restore_memory(frozen['memory']) if mode == 'warm' else None)
                log = agent.run(body, args.budget)
                passed = body.pass_goal()
                r = record_log(fixture, log, seed, name, case)
                r.update(success=passed, opened=log.reached_goal, entered_exit=passed,
                         actions_to_success=log.n_actions if passed else None,
                         recipe_actions=agent.replay_actions, repair_reason=agent.repair_reason,
                         moves=maze.steps, known_cells=len(body.navigator.memory.cells),
                         navigation=body.navigation)
                explain_record(r, log, likelihood)
                modes[mode] = r
            modes['no_recipe'] = copy_record(modes['cold'])
            fixture = scene(seed, name, case)
            try:
                original = run_key_chain(MazeWorld(seed=seed, size=args.size), fixture, seed=seed,
                                         move_budget=args.moves, fixture_budget=args.budget)
                modes['scripted'] = {'success': original.success, 'actions': original.fixture_actions,
                                 'actions_to_success': original.fixture_actions if original.success else None,
                                 'moves': original.moves, 'case': case, 'seed': seed, 'type': name,
                                 'events': fixture.events, 'script_reports': original.events}
            except ValueError as exc:
                modes['scripted'] = {'success': False, 'actions': 0, 'actions_to_success': None,
                                     'moves': 0, 'case': case, 'seed': seed, 'type': name,
                                     'invalid_layout': str(exc)}
            rows.append({'seed': seed, 'case': case, 'type': name, 'modes': modes})
            print(f'situated {seed} {case}: ' + ', '.join(f"{m} {r['success']}/{r['actions']} moves={r['moves']}"
                                                        for m, r in modes.items()), flush=True)
    assert source_hashes() == frozen['sources']
    return {'kind': 'fresh_situated_mock_gate', 'sources': sources,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'summary': {mode: summarize([r['modes'][mode] for r in rows]) for mode in rows[0]['modes']},
            'episodes': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--layouts', type=int, default=4)
    p.add_argument('--size', type=int, default=15)
    p.add_argument('--moves', type=int, default=30000)
    p.add_argument('--budget', type=int, default=800)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
