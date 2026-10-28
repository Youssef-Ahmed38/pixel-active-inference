# The G1 delivers parcels by bicycle in a town it has never seen

<img src="media/bike/delivery.gif" width="720">

`media/bike/delivery.mp4`: a delivery in town 4549 (one of the final evaluation towns) from the depot
to house 2. Top left the chase camera; top right the G1's head camera (it turns toward the street it
is checking); bottom right the rider's own map (its route, the streets it has seen clear, the barriers
and parked cars it has seen; nothing it has not seen); bottom left its speed, roll, target lean and
steering, and every look with what the camera saw and what it decided.

## Results (final towns 4540-4559, never used before; 40 deliveries)

| Rider (same towns, same houses, same steering) | Delivered (Wilson 95%) | Collisions | Falls | Left the road |
|---|---:|---:|---:|---:|
| **Looking and replanning** | **36/40** [76.9, 96.0] | **0** | 2 | 2 |
| Blind (shortest route on the map, no looking) | 21/40 [37.5, 67.1] | 19 (12 barriers, 7 cars) | 0 | 0 |

Paired over the 40 deliveries: 15 delivered only by the looking rider, 0 only by the blind one
(sign test p = 0.0001). The head camera read 178 of 178 streets correctly against the hidden truth
(17 replans, many lane changes). Delivered rides: 45.8 s and 89.7 m on average (the detours), mean
heading error 3.7 degrees, mean distance from the lane line 0.14 m. All four failures are one case:
a barrier on a side street seen only 5 m before the intersection that leaves a U-turn as the only way
(three at the town's corner intersection); the loop then starts too sharply and the rider falls or
runs wide. `results/bike_g1_delivery.json`.

    python scripts/bike_g1_delivery_eval.py --towns 4540 4560 --output results/bike_g1_delivery.json
    python scripts/bike_g1_delivery_video.py --town 4549 --house 2 --out docs/media/bike/delivery

**Earlier evaluations (kept, each on its own new towns):** after each, the failures were studied and
the next evaluation used towns never seen before.

| Evaluation | Rider | Looking delivered | Blind delivered | Failures of the looking rider |
|---|---|---:|---:|---|
| 1: towns 4500-4519 | v2 steering; no U-turn manoeuvre; looks 3 m before every intersection | 33/40 | 24/40 | 6 left the road and 1 fell, all where a U-turn was needed (the path reversed on the spot) |
| 2: towns 4520-4539 | v3 steering with U-turn loops | 34/40 | 20/40 | 4 falls, 2 off the road: barriers straight ahead seen only 4 m before the intersection, too late to turn |
| 3 (final): towns 4540-4559 | v3; the street straight ahead is checked from 10 m | 36/40 | 21/40 | 4 late U-turns (above) |

`results/bike_g1_delivery_first.json`, `results/bike_g1_delivery_second.json`.

## Learning to corner by practice

`scripts/bike_g1_turn_practice.py`: routes of 3-4 streets with at least one 90-degree corner, in
practice towns 4000-4003 (2 routes each); a cross-entropy search over the steering law's gains keeps
only improvements of the practice score; development towns 4100-4104 only check; each final check on
its own new towns. A ride ends with a fall, or when the rider is more than 3 m from its route.

| Stage | What was practised | Final towns | Practised | Before this practice | Untuned law |
|---|---|---|---:|---:|---:|
| v1 | progress along the route | 4200-4209 | 20/20 routes; heading error 24.8 deg, 0.99 m from the lane (max 2.95 m) | - | 0/20 |
| v2 | progress minus 0.3 x distance from the lane, from v1 | 4210-4219 | **20/20; heading error 3.9 deg, 0.14 m (max 0.38 m)** | v1: 19/20, 0.99 m | 0/20 |
| v3 | as v2, plus 2 U-turn routes per town and a U-turn speed, from v2 | 4220-4229 | **40/40 including 20 U-turns**; 5.8 deg, 0.16 m (max 0.91 m) | v2: 20/40 (falls in every U-turn) | 0/40 |

v1 rode without falling but drifted wide in corners and wove back (it was scored on progress only);
adding the distance from the lane to the score (as CycleRL rewards tracking) made it ride the lane.
Many final routes are the same shape in different towns (the grid repeats), so the number of truly
different routes is smaller than the count. `results/bike_g1_turns_v1.json`, `_v2.json`, `_v3.json`.

    python scripts/bike_g1_turn_practice.py --output results/bike_g1_turns_v1.json
    python scripts/bike_g1_turn_practice.py --start-from results/bike_g1_turns_v1.json --track-weight 0.3 --iterations 10 --final-towns 4210 4220 --output results/bike_g1_turns_v2.json
    python scripts/bike_g1_turn_practice.py --start-from results/bike_g1_turns_v2.json --u-turns 2 --track-weight 0.3 --iterations 12 --final-towns 4220 4230 --output results/bike_g1_turns_v3.json

(The v1 file was produced before the `--start-from`, `--track-weight` and `--u-turns` options existed;
its command is the first line with those options at their defaults.)

## The task

`pai/bike/town.py` builds a town in MuJoCo: a 4 x 4 grid of intersections 25 m apart (24 streets),
asphalt streets 4 m wide with a dashed centre line, pavements, buildings on every block, a blue depot
pad and eight numbered houses with orange drop-off pads and posts. Obstacles stand on about 45% of the
streets (none next to the depot):

- **barriers** (a white barrier and cones across the street, about 20% of streets): the street cannot
  be passed;
- **parked cars** (one lane, about 25%): passable in the other lane.

A delivery starts at the depot and ends at a house's pad. The rider is given the **street map** (the
intersections and streets, as a map app would) but **not the obstacles**: they are hidden state it can
only learn by looking. The obstacles are drawn for the camera; contact with them is checked
geometrically (inside a barrier's line or a car's footprint plus 0.3 m is a collision).

## What the rider does (and what is given or learned)

| Part | How | Given or learned |
|---|---|---|
| Steering and cornering | `pai/bike/riding.py` `LeanFollower`: lean into the curve ahead (the lean a steady turn needs, plus a heading correction), steer to reach and hold that lean (the countersteer comes out by itself), slow before corners; the lane is rounded to 3.4 m at corners; a U-turn is a 2.4 m loop round the intersection | the law's form is given; its 9 gains and speeds are **learned by practice** (`scripts/bike_g1_turn_practice.py`) |
| Seeing obstacles | `pai/bike/navigation.py` `inspect_street`: before each intersection the G1 turns its head toward the street it plans to take (10 m before when it is straight ahead, 4 m when it is around a corner, where the corner buildings hide it until then); the head camera's depth image becomes 3-D points, and points standing 0.2-2 m above the road inside that street are an obstacle: a barrier if they span more than 2.3 m of the street, else a parked car, placed in the lane its points are in and taken to be about 4 m long | the reading rule is given |
| Belief and route | a belief P(blocked) per street, starting from the rate of blocked streets **counted in the practice towns** (0.125); routes minimise length + P(blocked) x 60 m and are replanned whenever a look changes the belief, avoiding a U-turn when another way exists; after a replan the new street is checked at once | the prior is counted; the planner is given |
| Parked cars | the planned lane path moves into the other lane past a car in the rider's lane (smooth 3.5 m ramps), and keeps 0.5 m more room from a car in the other lane | given |

**Stated simplifications:** the head turn is a camera direction, not a G1 neck motion (the G1 has no
neck joint); the drive torque is applied at the rear wheel and the steering is a servo on the fork
held by the G1's hands (as in `docs/BIKE_G1.md`); for cornering the G1's arm servos are softened so the
arms follow the handlebar, and the steering servo is stronger (with the stiff arms of the
straight-riding work the handlebar could not turn past about 0.2 rad, too little for a corner);
there is no other traffic.

## Findings on the way

- The first cornering attempt (the straight-riding balance rule plus pure-pursuit steering) stalled
  at 66% of the practice routes, and its "final" result looked like 4/20 completed routes with no
  falls. That was reward hacking: the progress measure took the nearest path point within the
  next 40, so a rider far off the route still "advanced" (mean cross-track error 49 m, max 186 m). The
  measure now ends a ride that leaves the route by more than 3 m; that result is discarded
  (`data/bike_g1_turns_linear_invalid.json`, local).
- The rule's form was the problem: steering toward the target heading makes a bicycle lean outward.
  Leaning first and steering to hold the lean (how riders corner) fixed it.
- With the G1's arm servos as stiff as in the straight riding, the handlebar could not turn past about
  0.2 rad, too little for a 3.4 m corner; with soft arms and a stronger steering servo it reaches
  about 0.52 rad.
- Camera reading was checked first standing at the approach to every street of two test towns (4300,
  4301; all approach directions): house posts on the road edge were read as parked cars (they were
  moved onto the pavement); barriers seen at an angle could look narrower than a street (threshold
  2.3 m; a car covers at most 1.85 m); parked cars seen at an angle show mostly their near side, close
  to the centre line, so their lane is taken from the side their points are on.
- Bugs found in the test town before the evaluations: after a replan the rider did not look down its
  new street (it rode into a parked car on the detour); the lane change was decided per path point,
  which misfired on curves.

## How this compares with published work

| Work | Robot | Navigation | Obstacles | Result |
|---|---|---|---|---|
| CycleRL (Liu et al., Sun Yat-sen University, arXiv 2603.15013, 2026) | a bicycle robot (no rider), Isaac Sim and hardware | heading and speed commands; PPO | none | balance success 99.9% sim, 95% real; heading error 1.15 deg sim, 2.04 deg real; names vision-based navigation as future work |
| Scooter-driving humanoid (MDPI Sensors, 2026) | real full-size humanoid on a two-wheeled scooter | operator heading and speed; deep RL | a 60 mm bump | 0.96 straight, 0.92 light turns, 0.72 sharp turns |
| Ultra Mobility Vehicle (2026) | bicycle robot | RL (PPO, Isaac Lab) | jumps onto tables | driving 1.00, jumps 0.94, flips 0.97 |
| Randlov and Alstrom (ICML 1998) | simulated bicycle, no body | ride to a goal ~1 km away; Sarsa(lambda) | none | reaches the goal after learning |
| **This work** | **G1 humanoid on a realistic bicycle (simulation)** | **street map + head camera, belief over blocked streets, replanning; corners and U-turns** | **barriers and parked cars, unknown in advance** | **36/40 deliveries in new towns vs 21/40 blind; 0 collisions; heading error 3.7 deg, 0.14 m from the lane** |

Where this goes past the published work: a humanoid rider on a bicycle that plans a delivery route
in a town, discovers obstacles with its own head camera and replans (CycleRL's stated next step).
Where it is behind: real hardware; heading accuracy (CycleRL 1.15 deg in simulation on straight
heading commands, here 3.7 deg over town routes with corners); a richer learned policy (CycleRL's
PPO) instead of a law with 9 gains learned by practice; the arms do not produce the steering torque;
U-turns decided late; no other traffic.

## Bicycle v2: late U-turns, a learned riding policy (PPO), and robustness

### Fixing the late U-turns
The four failures of the final delivery evaluation all ended in a U-turn decided about 5 m before an
intersection. Tracing them found two causes:

- **A bug.** After switching to the U-turn route, the rider located itself on the nearest point of
  the new path, which was on the U-turn's way back in the other lane, 2.2 m away. It then believed it
  faced 126 degrees off its route and threw itself into a turn. It now takes the nearest path point
  that runs the way it faces.
- **Dead ends.** A barrier on the only side street of an intersection (three were at the town's
  corner) leaves a late U-turn as the only way on. Routes now include the expected cost of being
  forced into one: at each intersection, the probability that every way on is blocked (from the
  belief, e.g. 12.5% at a corner with one unchecked exit, about 1.6% at an edge intersection with
  two), times 150 m. With both changes, 4 of the 6 failed deliveries of the earlier evaluations were
  delivered (development use of those towns); the two left are true dead ends.

**The practised rule is brittle.** The practised lean-target rule (v3) rides the exact U-turn loop
it practised (18/18 U-turns on practice and development towns) but only 9/18 when the loop's centre
moves 0.3 m, 0/18 at 1.0 m, and 0/18 with a 2.8 m instead of 2.4 m loop: nine gains tuned on one
geometry do not generalise to another.

### Three training seeds of the practised rule
The cornering practice chain (v1 progress, v2 lane tracking, v3 with U-turns) was repeated with two
new search seeds (`--seed 1`, `--seed 2`) and every resulting v3 rider ridden on the same 40 new
routes (towns 4240-4249, 2 with corners and 2 with a U-turn per town):

| Training seed | Completed | Falls | Heading error | Distance from the lane, mean / max |
|---|---:|---:|---:|---:|
| 0 (the published rider) | 40/40 | 0 | 5.4 deg | 0.15 / 0.91 m |
| 1 | 40/40 | 0 | 4.2 deg | 0.18 / 0.65 m |
| 2 | 40/40 | 0 | 8.8 deg | 0.18 / 0.87 m |
| mean +- SD | 120/120 | 0 | 6.1 +- 2.4 deg | 0.17 +- 0.02 m |

The final rider does not depend on a lucky practice run; the intermediate stages do vary (seed 1's
v2 fell on 7 of 20 routes before the U-turn stage). `results/bike_seeds/`.

    python scripts/bike_g1_turn_practice.py --seed 1 --final-towns 4240 4250 --output results/bike_seeds/turns_v1_seed1.json
    python scripts/bike_g1_turn_practice.py --seed 1 --start-from results/bike_seeds/turns_v1_seed1.json --track-weight 0.3 --iterations 10 --final-towns 4240 4250 --output results/bike_seeds/turns_v2_seed1.json
    python scripts/bike_g1_turn_practice.py --seed 1 --start-from results/bike_seeds/turns_v2_seed1.json --u-turns 2 --track-weight 0.3 --iterations 12 --final-towns 4240 4250 --output results/bike_seeds/turns_v3_seed1.json

### A learned riding policy (PPO, as CycleRL)
`pai/bike/ppo_rider.py`, `scripts/bike_g1_ppo_train.py`: the policy sees the bicycle's IMU, speed
and steering, its distance from the lane, the heading to route points 1, 2.5 and 4 m ahead and the
curvature ahead (16 numbers), and outputs steering and drive torque. 512 riders on random town routes
(30% with a U-turn) in training towns 5000-5007, simulated on CPU threads (`mujoco.rollout`), the
network on the GPU.

- **From scratch (negative).** The first reward paid route progress anywhere, and the policy learned
  to balance while drifting off the route. With progress counted only near the lane it learned to
  balance and keep the lane (0.25 m) but not to lean into corners: on development towns it completed
  **0/20** routes at every checkpoint (best mean progress 38%), after about 12M steps. CycleRL trains
  at far larger scale.
- **Residual PPO.** The policy corrects the practised rule by at most 0.15 rad of steering and 4 N m of
  torque, trained with random corner (3.0-3.8 m) and U-turn (2.2-2.6 m) radii, i.e. exactly where the
  rule is brittle. With zero correction it rides exactly as the rule.

**Final routes** (new towns 4230-4239: 2 routes with corners and 2 with a U-turn per town, 40
routes; checkpoints chosen on development towns 4100-4104 only: residual iteration 600 by most
completions then least distance from the lane, from-scratch iteration 200 by most progress):

| Rider | Standard geometry | Random corner and U-turn radii | Mean / max distance from the lane (random) |
|---|---:|---:|---:|
| **Residual PPO** | **40/40** [91.2, 100] | **40/40** [91.2, 100] (U-turns 20/20) | **0.08 / 1.29 m** |
| Practised rule (v3) | 40/40 | 31/40 [62.5, 87.7] (U-turns 11/20; 9 falls) | 0.15 / 1.03 m |
| From-scratch PPO | 0/40 | 0/40 (all fall, mean progress 31%) | - |

With random geometry the residual policy completed 9 routes the rule failed and never the reverse
(paired sign test p = 0.004); on the standard geometry both complete everything, and the residual
rides about twice as close to the lane (0.08 vs 0.15 m). `results/bike_g1_ppo_final_{standard,varied}.json`,
development selection `results/bike_g1_ppo_dev_{scratch,residual}.json`.

**Final deliveries** (new towns 4560-4579, 40 deliveries; both riders with the U-turn fixes):

| Steering of the looking rider | Delivered | Collisions | Falls | Blind rider, same towns | Paired vs blind |
|---|---:|---:|---:|---:|---|
| **Practised rule** | **37/40** [80.1, 97.4] | 1 | 2 | 23/40 (17 collisions) | 14-0, p = 0.0001 |
| Residual PPO | 34/40 [70.9, 92.9] | 1 | 4 (+1 off the road) | 23/40 | 11-0, p = 0.001 |

The residual policy rides better but does not deliver better (rule-only 4, residual-only 1,
p = 0.38): five of its six failures are lane changes around parked cars, often just after a corner,
which its training never contained (no cars). The practised rule stays the delivery default.

**Practising lane changes did not help (development only).** A second residual run started from the
selected policy and added parked cars to half of its training routes (the path changes lane around
1-2 cars, often just after a corner; the rule alone falls on 11 of 27 such routes), with random
geometry, 40M steps. On development deliveries in new towns 4580-4589 (20 deliveries) every
checkpoint of it delivered 15/20 (2 collisions with parked cars), the first residual 17/20, the
practised rule 19/20 (no collision). Worse on development, so not taken to a final; the rule stays
the delivery rider (`results/bike_g1_ppo_dev_deliveries_run2.json`). The
first rule run of this final stopped on an OpenGL error of the head camera (no result written) and
was run again unchanged. `results/bike_g1_delivery_v2_{rule,residual}.json`.

    python scripts/bike_g1_ppo_train.py --out data/bike_ppo/run2 --hours 3 --init <run1 iteration 150>   # from scratch
    python scripts/bike_g1_ppo_train.py --out data/bike_ppo/residual1 --hours 2.5 --residual results/bike_g1_turns_v3.json --vary-geometry
    python scripts/bike_g1_ppo_eval.py --ckpts results/bike_ppo/residual_selected.pt results/bike_ppo/scratch_selected.pt --towns 4230 4240 --vary-geometry --with-rule --output results/bike_g1_ppo_final_varied.json
    python scripts/bike_g1_delivery_eval.py --towns 4560 4580 --ppo results/bike_ppo/residual_selected.pt --output results/bike_g1_delivery_v2_residual.json
    python scripts/bike_g1_delivery_eval.py --towns 4560 4580 --output results/bike_g1_delivery_v2_rule.json

### Faster training without a GPU simulator
MJX (MuJoCo on the GPU, via WSL2) was no faster for the G1 on its bicycle on this laptop (about 5,200
control steps/s at 4,096 parallel riders, the same as CPU threads), and MuJoCo Warp ran out of the
8 GB of GPU memory. The bottleneck was the per-rider Python code; computing observations, progress,
rewards and the residual rule for all riders as arrays (`FastBikeVecEnv`, identical numbers) is 2.5x
faster.

## Audit: are the perception and decisions real?

`scripts/bike_g1_audit.py`, development towns 4300-4304 (2 deliveries each), with the **truth guard**
on: the town's hidden obstacles can only be read inside `pai/bike/town.py:scoring()` (counting
practice-town outcomes, the world's collision check, and logging each camera reading against the
truth); any other read raises `TruthAccessError` (tested).

| Test | Delivered | Collisions | What it shows |
|---|---:|---:|---|
| Normal rider | 10/10 | 0 | |
| **Blindfold** (the head camera sees nothing) | 5/10 | 5 | the same as the blind rider (5/10, 5 collisions); all 29 of its readings were "clear": its success comes from what it sees |
| **Tamper** (the camera sees this town's obstacles; the hidden truth is swapped for another town's on the same streets) | 6/10 | 4 | its 38 readings match what was shown 38/38 and the hidden truth 25/38; on the 13 streets where they differ, all 13 match what was shown: decisions follow the camera |
| **Determinism** (each delivery ridden twice) | 10/10 identical | | outcome, time, route and every reading are reproducible |

`results/bike_g1_audit.json`.

    python scripts/bike_g1_audit.py --towns 4300 4305 --output results/bike_g1_audit.json
