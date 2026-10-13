"""Maze sensing, memory and paired exploration gate."""

from pai.maze import Explorer, MazeWorld, RandomWalker, run_episode
from pai.maze.agent import PlaceMemory


def test_local_observation_hides_distant_goal_and_walls_behind_walls():
    world = MazeWorld(seed=4, size=11, sensor_range=2)
    obs = world.observe()
    assert world.goal not in obs.visible or abs(world.goal[0] - 1) + abs(world.goal[1] - 1) <= 2
    assert len(obs.visible) < world.grid.size
    assert all(abs(p[0] - 1) + abs(p[1] - 1) <= 2 for p in obs.visible)


def test_memory_routes_only_through_observed_free_cells():
    world = MazeWorld(seed=2, size=11)
    memory = PlaceMemory()
    memory.update(world.observe())
    free = [p for p, kind in memory.cells.items() if kind == "floor" and p != world.start]
    assert free
    route = memory.route(world.start, free[0])
    assert route and route[-1] == free[0]
    assert all(memory.cells[p] == "floor" for p in route)
    assert not memory.route(world.start, (999, 999))


def test_explorer_beats_random_on_paired_mazes():
    explorer, random = [], []
    for seed in range(8):
        e_world = MazeWorld(seed, size=15)
        r_world = MazeWorld(seed, size=15)
        explorer.append(run_episode(e_world, Explorer(size=15, seed=seed), budget=90)["coverage"])
        random.append(run_episode(r_world, RandomWalker(seed=seed), budget=90)["coverage"])
    assert sum(explorer) / len(explorer) > sum(random) / len(random) + 0.10


def test_goal_search_uses_discovered_goal():
    world = MazeWorld(seed=7, size=11)
    result = run_episode(world, Explorer(size=11, seed=7), budget=180, stop_at_goal=True)
    assert result["found_goal"]
    assert result["goal_step"] <= 180


def test_full_completion_gate_on_paired_seeds():
    for seed in range(8):
        world = MazeWorld(seed=1000 + seed, size=15)
        result = run_episode(world, Explorer(size=15, seed=1000 + seed), budget=300)
        assert result["found_goal"], seed
        assert result["visited"] == result["reachable"], seed
