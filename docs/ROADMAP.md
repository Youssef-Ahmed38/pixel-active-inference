# Vision: an agent that thinks through perception

## The idea

Brains are thought to work by prediction. Fast sensory levels predict what the senses will report;
slower levels explain the prediction errors by inferring the hidden causes behind them; the slowest
levels hold goals. Surprise is not only an error: it tells the agent where to look, what to touch and
what to learn next (Rao and Ballard 1999; Friston 2008, 2010).

This project builds an embodied agent organised that way and asks two questions:

1. **Can one design, unchanged in its principles, carry across very different bodies and worlds?**
   A robot arm, a Unitree G1 humanoid that walks, explores mazes through its own head camera and
   rides a bicycle through towns, and mechanism, chess and bilingual-language worlds.
2. **Does the cause level make the agent better at what matters when the world changes?** Noticing
   that something is wrong, naming what it is, repairing its behaviour, and explaining itself.

## The agent: three levels

| Level | Speed | What it does |
|---|---|---|
| **Sensory** | fast, every step | a learned model predicts the next observation (camera, depth, touch, force, body state); its precision-weighted error is the surprise |
| **Cause** | slower | infers hidden causes from surprise and estimates their parameters: a push, a heavier or slippery object, a moved camera, a blocked street, a misread wall, the reason for a fall |
| **Goal** | slowest | chooses plans by expected free energy: reach the goal, and take actions whose only purpose is to find out (look, touch, check a street) |

Episodes are remembered with their causes and responses; explanations in English and Arabic are read
from the agent's own beliefs, never from the simulator's hidden state.

## What the evidence says so far (final seeds never used in development)

- **The cause level makes behaviour repairable and explainable.** The arm names hidden causes in
  88% of new episodes (97/100 with learned likelihoods) and, by recalibrating after a camera shift,
  succeeds in 90% instead of 20%; the humanoid names the cause of its own falls in 92.6%; explanations
  in two languages come only from the agent's beliefs and are scored against hidden truth.
- **The information term matters where the world is ambiguous.** Curiosity over routes finds goals in
  84 of 100 branchy mazes against 67 for nearest-frontier exploration; under persistent sensor errors,
  learning that repeated looks carry no information and touching doubtful walls lifts the goal rate
  from 6-13% to 49%; a humanoid that looks down streets with its own camera and replans delivers parcels
  in 37 of 40 new towns against 23 riding blind.
- **Perception is real, and audited.** On the bicycle a blindfolded camera reduces the rider to the
  blind rider's results, a camera shown different obstacles from the hidden truth follows what it sees
  on every street where the two differ, rides are reproducible, and the rider's code cannot read the
  hidden obstacles at all.
- **What is not yet learnable.** A registered test asked whether the cause level can be discovered from
  perception without labels: a three-level hierarchy did no better than a flat model of the same size
  (48.7% vs 44.5% cluster accuracy, against 90.5% for the explicit cause level). The structure of the
  causes is, for now, given. Learning it is the central open problem.

## How we work

- **Honest evaluation:** development seeds for all tuning; frozen parameters; final seeds run once;
  Wilson intervals and paired tests; several training seeds where training is involved.
- **Failures are results:** bugs that removed earlier findings, training that forgot skills, fixes
  that did not work and pre-registered negative tests stay in the reports.
- **Learned versus given** is stated for every component.
- **One laptop:** physics on the CPU threads, networks and GPU-friendly physics (MJX / MuJoCo Warp) on
  the GPU (RTX 5060, 8 GB), with every GPU-trained policy verified on the exact CPU physics.

## Next

1. **Learning the cause structure from perception:** the open problem above, now with a data protocol
   that can test it.
2. **Standard baselines and training seeds** in every world: model-free and recurrent PPO for the arm,
   the published GPU walking recipe (MuJoCo Playground) for the humanoid, PPO for the bicycle.
3. **A more physical rider:** the G1's arms producing the steering, and other traffic in the towns.
4. **The paper:** the agent as a perceptual hierarchy, with the cause level as its contribution and the
   learned-hierarchy test as a registered negative result.
