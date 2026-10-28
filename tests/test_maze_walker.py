"""Solid maze walls for the learned G1 walker (needs the G1 assets and the released policy)."""
import numpy as np
import pytest

from pai.maze.env import MazeWorld

walker = pytest.importorskip("pai.maze.walker")
if not walker.POLICY.exists():
    pytest.skip("released walking policy not available", allow_module_level=True)

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def neighbours(world):
    r, c = world.start
    return {d: bool(world.grid[r + d[0], c + d[1]]) for d in DIRECTIONS}


def test_wall_contacts_use_only_the_new_bit():
    world = MazeWorld(0)
    m = walker.build_maze_body(world, solid=True).model
    walls = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("wall_")]
    assert m.body_contype[m.body("pelvis").id] == walker.WALL_BIT   # the broad phase sees the bit
    body = [i for i in range(m.ngeom) if m.geom_contype[i] == walker.WALL_BIT]
    assert walls and len(body) == 23   # every capsule the G1 pairs with the floor
    floor = m.geom("floor").id
    for i in body:
        # no new contacts with the floor or between body parts: only walls listen to this bit
        assert not (m.geom_contype[i] & m.geom_conaffinity[floor]) and not (m.geom_contype[floor] & m.geom_conaffinity[i])
        assert m.geom_conaffinity[i] == 0
    assert all(m.geom_conaffinity[w] == walker.WALL_BIT and m.geom_contype[w] == 0 for w in walls)
    visual = walker.build_maze_body(world, solid=False).model
    assert not any(visual.geom_contype[i] & walker.WALL_BIT for i in range(visual.ngeom))


def test_walking_into_a_solid_wall_is_stopped_and_felt():
    world = MazeWorld(0)
    wall_dir = next(d for d, wall in neighbours(world).items() if wall)
    w = walker.WalkerMaze(world, seed=0, solid=True)
    start_xy = w.xy()
    deepest = []
    w.on_frame = lambda env: deepest.append(float(np.linalg.norm(env.qpos()[0, :2] - start_xy)))
    obs = w.step(wall_dir)
    assert not w.fell
    assert w.bumps == 1 and obs.collided
    assert world.position == world.start
    # the wall face is 0.5 m away: the pelvis is held back by the legs and arms hitting it, while
    # with visual walls it only stops once it is inside the wall cell (beyond 0.5 m)
    assert max(deepest) < 0.45
    v = walker.WalkerMaze(MazeWorld(0), seed=0, solid=False)
    visual = []
    v.on_frame = lambda env: visual.append(float(np.linalg.norm(env.qpos()[0, :2] - start_xy)))
    v.step(wall_dir)
    assert max(visual) > 0.5


def open_world():
    """Maze 0 with every wall removed: solid and visual differ only in the collision masks."""
    world = MazeWorld(0)
    world.grid[:] = 0
    return world


def test_open_floor_walking_is_unchanged_by_solid_walls():
    world = open_world()
    open_dir = (1, 0)
    target = walker.cell_xy((world.start[0] + open_dir[0], world.start[1] + open_dir[1]), world.start, 1.0)
    paths = []
    for solid in (False, True):
        w = walker.WalkerMaze(open_world(), seed=0, solid=solid)
        trace = []
        w.on_frame = lambda env, trace=trace: trace.append(env.qpos()[0].copy())
        w._walk_to(target)
        paths.append(np.array(trace))
    assert paths[0].shape == paths[1].shape
    assert np.allclose(paths[0], paths[1], atol=1e-9)


def test_timed_out_walk_is_not_a_touch():
    world = MazeWorld(0)
    open_dir = next(d for d, wall in neighbours(world).items() if not wall)
    w = walker.WalkerMaze(world, seed=0, step_timeout_s=0.1)
    obs = w.step(open_dir)
    assert world.position == world.start and not obs.collided
    assert world.events[-1]["type"] == "walk_failed"


def test_noisy_sensor_sits_on_top_of_the_walker():
    from pai.maze.noisy import NoisySensorMaze
    world = MazeWorld(0)
    wall_dir = next(d for d, wall in neighbours(world).items() if wall)
    env = NoisySensorMaze(walker.WalkerMaze(world, seed=0), seed=0, persistent=True)
    assert env.size == world.size and env.position == world.start
    obs = env.step(wall_dir)
    # the body hit the wall: touch reports it, whatever the camera said
    assert obs.touched == (world.start[0] + wall_dir[0], world.start[1] + wall_dir[1])
    assert obs.visible[obs.touched] == "wall"
