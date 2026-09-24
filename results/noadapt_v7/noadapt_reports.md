# Episode reports (the agent's own account vs ground truth)

```
Episode 0: goal on(red, plate): SUCCESS after 118 planning steps (11.8 s).
  Progress: above_object at 1.1 s, at_object at 7.7 s, grasped at 7.8 s, lifted at 8.5 s, over_target at 9.6 s, lowered at 11.1 s, opened at 11.3 s, released at 11.8 s.
  Surprise: my predictions failed at 26 steps, most at 10.0 s (surprise 141; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.8, extra_mass_kg=0.489, sag_mm=0.26).
  Ground truth: payload at 0.0 s (mass=0.4910885061964363).

Episode 1: goal on(red, plate): SUCCESS after 112 planning steps (11.2 s).
  Progress: above_object at 2.8 s, at_object at 5.8 s, grasped at 5.9 s, lifted at 6.5 s, over_target at 8.6 s, lowered at 10.5 s, opened at 10.7 s, released at 11.2 s.
  Surprise: my predictions failed at 40 steps, most at 9.0 s (surprise 84; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.77, extra_mass_kg=0.385, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.38093601412916106).

Episode 2: goal on(red, plate): SUCCESS after 96 planning steps (9.6 s).
  Progress: above_object at 1.1 s, at_object at 4.2 s, grasped at 4.3 s, lifted at 5.2 s, over_target at 6.7 s, lowered at 9.0 s, opened at 9.2 s, released at 9.6 s.
  Surprise: my predictions failed at 38 steps, most at 7.1 s (surprise 57; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.08, extra_mass_kg=0.314, sag_mm=0.04).
  Ground truth: payload at 0.0 s (mass=0.3122920571808584).

Episode 3: goal on(red, plate): SUCCESS after 75 planning steps (7.5 s).
  Progress: above_object at 1.5 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.3 s, over_target at 4.9 s, lowered at 6.8 s, opened at 7.0 s, released at 7.5 s.
  Surprise: my predictions failed at 35 steps, most at 5.3 s (surprise 55; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.02, extra_mass_kg=0.307, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.30495829065855873).

Episode 4: goal on(red, plate): SUCCESS after 133 planning steps (13.3 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 9.3 s, at_object at 10.4 s, grasped at 10.5 s, lifted at 11.0 s, over_target at 11.4 s, lowered at 12.7 s, opened at 12.9 s, released at 13.3 s.
  Surprise: my predictions failed at 52 steps, most at 8.0 s (surprise 172; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.38, extra_mass_kg=0.548, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.5439810717600817).

Episode 5: goal on(red, plate): SUCCESS after 68 planning steps (6.8 s).
  Progress: above_object at 1.3 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.4 s, over_target at 4.6 s, lowered at 6.1 s, opened at 6.3 s, released at 6.8 s.
  Surprise: my predictions failed at 27 steps, most at 5.1 s (surprise 187; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.66, extra_mass_kg=0.577, sag_mm=0.01).
  Ground truth: payload at 0.0 s (mass=0.5738266731833165).

Episode 6: goal on(red, plate): SUCCESS after 91 planning steps (9.1 s).
  Progress: above_object at 1.1 s, at_object at 4.5 s, grasped at 4.6 s, lifted at 5.5 s, over_target at 7.0 s, lowered at 8.5 s, opened at 8.7 s, released at 9.1 s.
  Surprise: my predictions failed at 30 steps, most at 7.7 s (surprise 135; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.75, extra_mass_kg=0.484, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.4819907327301539).

Episode 7: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.1 s, at_object at 7.0 s.
  Surprise: my predictions failed at 22 steps, most at 9.5 s (surprise 142; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=80, end=99, force_N=[0.0, -0.0, 5.1], offset_mm=[0.1, 0.2, -0.2]).
  Ground truth: payload at 0.0 s (mass=0.5188489682951996).

Episode 8: goal on(red, plate): SUCCESS after 61 planning steps (6.1 s).
  Progress: above_object at 1.5 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.3 s, over_target at 4.3 s, lowered at 5.5 s, opened at 5.7 s, released at 6.1 s.
  Surprise: my predictions failed at 22 steps, most at 4.7 s (surprise 122; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.55, extra_mass_kg=0.464, sag_mm=0.15).
  Ground truth: payload at 0.0 s (mass=0.46308749743962685).

Episode 9: goal on(red, plate): SUCCESS after 109 planning steps (10.9 s).
  Progress: above_object at 1.1 s, at_object at 7.0 s, grasped at 7.1 s, lifted at 7.9 s, over_target at 9.0 s, lowered at 10.3 s, opened at 10.5 s, released at 10.9 s.
  Surprise: my predictions failed at 24 steps, most at 9.7 s (surprise 193; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.73, extra_mass_kg=0.584, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.5805217271363304).

Episode 10: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.1 s, at_object at 3.2 s, grasped at 3.5 s, lifted at 4.3 s, over_target at 5.7 s, lowered at 7.2 s, opened at 7.4 s, released at 7.7 s.
  Surprise: my predictions failed at 29 steps, most at 6.0 s (surprise 175; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.38, extra_mass_kg=0.548, sag_mm=0.1).
  Ground truth: payload at 0.0 s (mass=0.5447560662364597).

Episode 11: goal on(red, plate): SUCCESS after 60 planning steps (6.0 s).
  Progress: above_object at 1.0 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.8 s, over_target at 3.7 s, lowered at 5.3 s, opened at 5.5 s, released at 6.0 s.
  Surprise: my predictions failed at 25 steps, most at 4.0 s (surprise 53; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=2.95, extra_mass_kg=0.301, sag_mm=0.17).
  Ground truth: payload at 0.0 s (mass=0.30082155005104444).

Episode 12: goal on(red, plate): SUCCESS after 57 planning steps (5.7 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.5 s, over_target at 3.5 s, lowered at 5.1 s, opened at 5.3 s, released at 5.7 s.
  Surprise: my predictions failed at 26 steps, most at 4.0 s (surprise 178; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.45, extra_mass_kg=0.556, sag_mm=0.13).
  Ground truth: payload at 0.0 s (mass=0.5572212829762708).

Episode 13: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.3 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 3.6 s, over_target at 5.5 s, lowered at 7.1 s, opened at 7.3 s, released at 7.7 s.
  Surprise: my predictions failed at 35 steps, most at 6.3 s (surprise 54; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.07, extra_mass_kg=0.313, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.3100756725916393).

Episode 14: goal on(red, plate): SUCCESS after 68 planning steps (6.8 s).
  Progress: above_object at 1.4 s, at_object at 2.2 s, grasped at 2.7 s, lifted at 3.4 s, over_target at 4.8 s, lowered at 6.0 s, opened at 6.4 s, released at 6.8 s.
  Surprise: my predictions failed at 26 steps, most at 5.3 s (surprise 159; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.13, extra_mass_kg=0.523, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.5188966339289832).

Episode 15: goal on(red, plate): SUCCESS after 69 planning steps (6.9 s).
  Progress: above_object at 1.6 s, at_object at 2.6 s, grasped at 2.9 s, lifted at 3.6 s, over_target at 5.0 s, lowered at 6.3 s, opened at 6.5 s, released at 6.9 s.
  Surprise: my predictions failed at 27 steps, most at 5.5 s (surprise 73; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.49, extra_mass_kg=0.356, sag_mm=0.05).
  Ground truth: payload at 0.0 s (mass=0.3526966861807677).

Episode 16: goal on(red, plate): SUCCESS after 124 planning steps (12.4 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 7.6 s, at_object at 8.6 s, grasped at 8.7 s, lifted at 9.4 s, over_target at 9.8 s, lowered at 11.8 s, opened at 12.0 s, released at 12.4 s.
  Surprise: my predictions failed at 48 steps, most at 6.4 s (surprise 181; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.5, extra_mass_kg=0.561, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.5589536767049659).

Episode 17: goal on(red, plate): SUCCESS after 60 planning steps (6.0 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.8 s, over_target at 4.2 s, lowered at 5.4 s, opened at 5.6 s, released at 6.0 s.
  Surprise: my predictions failed at 26 steps, most at 4.8 s (surprise 123; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.56, extra_mass_kg=0.465, sag_mm=0.0).
  Ground truth: payload at 0.0 s (mass=0.4624383660747275).

Episode 18: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.0 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 3.5 s, over_target at 4.6 s, lowered at 5.8 s, opened at 6.0 s, released at 6.4 s.
  Surprise: my predictions failed at 23 steps, most at 5.0 s (surprise 91; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.81, extra_mass_kg=0.388, sag_mm=0.12).
  Ground truth: payload at 0.0 s (mass=0.3899135671612154).

Episode 19: goal on(red, plate): SUCCESS after 66 planning steps (6.6 s).
  Progress: above_object at 1.4 s, at_object at 2.3 s, grasped at 2.4 s, lifted at 3.1 s, over_target at 4.4 s, lowered at 5.9 s, opened at 6.1 s, released at 6.6 s.
  Surprise: my predictions failed at 28 steps, most at 5.0 s (surprise 104; spikes start at 21).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.21, extra_mass_kg=0.43, sag_mm=0.01).
  Ground truth: payload at 0.0 s (mass=0.4268061663592975).

Episode 20: goal on(red, plate): SUCCESS after 142 planning steps (14.2 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 10.5 s, at_object at 11.2 s, grasped at 11.3 s, lifted at 12.0 s, over_target at 12.3 s, lowered at 13.5 s, opened at 13.8 s, released at 14.2 s.
  Surprise: my predictions failed at 2 steps, most at 10.4 s (surprise 246; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=104, sink_mm=53.3, weight_lost_N=0.81).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 21: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 14.8 s, at_object at 12.7 s, grasped at 12.8 s, lifted at 13.4 s, over_target at 14.1 s.
  Surprise: my predictions failed at 5 steps, most at 3.8 s (surprise 191; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=38, sink_mm=35.5, weight_lost_N=0.77).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 22: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.3 s, at_object at 13.7 s, grasped at 13.8 s, lifted at 14.5 s, over_target at 14.8 s.
  Surprise: my predictions failed at 5 steps, most at 12.2 s (surprise 411; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=122, sink_mm=54.2, weight_lost_N=0.79).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 23: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 4 time(s) and went back to re-grasp it.
  Progress: above_object at 14.8 s, at_object at 12.4 s, grasped at 12.5 s, lifted at 13.0 s.
  Surprise: my predictions failed at 6 steps, most at 13.3 s (surprise 485; spikes start at 21).
  My explanation: something pushed my arm, probability 0.93 (onset=133, end=134, force_N=[0.0, 0.0, -0.4], offset_mm=[-0.2, 0.6, 0.4]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 24: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 2.0 s, at_object at 14.1 s, grasped at 14.2 s.
  Surprise: my predictions failed at 1 steps, most at 14.1 s (surprise 37; spikes start at 21).
  My explanation: something pushed my arm, probability 0.98 (onset=141, end=141, force_N=[1.0, 0.5, 2.1], offset_mm=[-0.1, -0.2, -0.4]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 25: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 14.1 s, at_object at 12.7 s, grasped at 12.8 s, lifted at 13.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened, probability 0.99.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 26: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.9 s, at_object at 13.9 s, grasped at 14.0 s, lifted at 14.7 s, over_target at 11.4 s.
  Surprise: my predictions failed at 2 steps, most at 4.9 s (surprise 162; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=49, sink_mm=27.6, weight_lost_N=0.75).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 27: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.2 s, at_object at 1.9 s.
  Surprise: my predictions failed at 1 steps, most at 1.9 s (surprise 31; spikes start at 21).
  My explanation: nothing unusual happened.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 28: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 8.4 s, at_object at 6.0 s, grasped at 6.1 s, lifted at 6.7 s.
  Surprise: my predictions failed at 2 steps, most at 4.1 s (surprise 278; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=41, sink_mm=35.5, weight_lost_N=0.8).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 29: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 4 time(s) and went back to re-grasp it.
  Progress: above_object at 13.7 s, at_object at 11.9 s, grasped at 12.0 s, lifted at 12.6 s, over_target at 10.6 s.
  Surprise: my predictions failed at 5 steps, most at 7.7 s (surprise 483; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=77, sink_mm=52.6, weight_lost_N=0.8).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 30: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 10.0 s, at_object at 6.0 s, grasped at 6.3 s, lifted at 7.2 s.
  Surprise: my predictions failed at 13 steps, most at 5.8 s (surprise 262; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=50, end=62, force_N=[-7.5, -0.4, -34.5], offset_mm=[0.4, 0.2, 0.4]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 31: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 9.8 s, grasped at 9.9 s, lifted at 10.7 s, over_target at 11.7 s.
  Surprise: my predictions failed at 10 steps, most at 8.5 s (surprise 407; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=85, sink_mm=50.3, weight_lost_N=0.83).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 32: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 13.5 s, at_object at 11.2 s, grasped at 11.3 s, lifted at 11.9 s.
  Surprise: my predictions failed at 4 steps, most at 6.4 s (surprise 94; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=64, end=64, force_N=[0.2, -4.1, -0.4], offset_mm=[-0.0, 0.1, 0.2]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 33: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 13.1 s, at_object at 14.0 s, grasped at 14.1 s, lifted at 14.8 s.
  Surprise: my predictions failed at 1 steps, most at 11.0 s (surprise 24; spikes start at 21).
  My explanation: nothing unusual happened, probability 0.99.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 34: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 2.2 s.
  Surprise: my predictions failed at 3 steps, most at 2.2 s (surprise 53; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=22, end=22, force_N=[-0.8, -0.2, 1.9], offset_mm=[-0.3, 0.2, -0.6]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 35: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 11.9 s, at_object at 9.2 s, grasped at 9.3 s, lifted at 9.9 s, over_target at 10.7 s.
  Surprise: my predictions failed at 4 steps, most at 2.9 s (surprise 84; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=29, end=30, force_N=[-0.0, 0.0, 0.2], offset_mm=[-0.3, 0.0, 0.2]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 36: goal on(red, plate): SUCCESS after 130 planning steps (13.0 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 8.6 s, at_object at 9.5 s, grasped at 9.6 s, lifted at 10.3 s, over_target at 10.7 s, lowered at 12.2 s, opened at 12.5 s, released at 13.0 s.
  Surprise: my predictions failed at 2 steps, most at 5.4 s (surprise 45; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=54, end=54, force_N=[-0.9, 0.2, 2.0], offset_mm=[-0.2, 0.2, -0.6]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 37: goal on(red, plate): SUCCESS after 135 planning steps (13.5 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 8.2 s, at_object at 10.0 s, grasped at 10.1 s, lifted at 10.8 s, over_target at 11.2 s, lowered at 12.8 s, opened at 13.1 s, released at 13.5 s.
  Surprise: my predictions failed at 2 steps, most at 8.1 s (surprise 387; spikes start at 21).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=81, sink_mm=53.0, weight_lost_N=0.83).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 38: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s.
  Surprise: my predictions failed at 2 steps, most at 3.5 s (surprise 33; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=34, end=36, force_N=[38.2, -21.6, -60.7], offset_mm=[-1.8, 1.3, -0.3]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 39: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 14.5 s, at_object at 11.1 s, grasped at 11.2 s, lifted at 11.9 s, over_target at 13.4 s.
  Surprise: my predictions failed at 3 steps, most at 9.2 s (surprise 4769; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=92, end=92, force_N=[1.7, 1.5, -36.4], offset_mm=[-0.0, 0.5, -1.3]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 40: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 13.4 s, at_object at 14.2 s, grasped at 14.3 s.
  Surprise: my predictions failed at 1 steps, most at 0.8 s (surprise 510; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=8, camera_offset_mm=[33.2, 6.0, 2.9], seen_jump=[-0.0332, -0.006, -0.0029]).
  Ground truth: camera_shift at 0.8 s (offset=[0.03319595248066462, 0.005969952775055404, 0.0029437902314850013]).

Episode 41: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 14.0 s, at_object at 6.1 s, grasped at 6.2 s, lifted at 3.3 s.
  Surprise: my predictions failed at 2 steps, most at 3.6 s (surprise 780; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=36, camera_offset_mm=[-30.9, -27.5, 10.4], seen_jump=[0.0309, 0.0275, -0.0104]).
  Ground truth: camera_shift at 3.6 s (offset=[-0.03107083983913775, -0.02752653498793452, 0.00994419871578422]).

Episode 42: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 11.7 s, at_object at 10.6 s, grasped at 10.7 s, lifted at 2.7 s, over_target at 4.3 s.
  Surprise: my predictions failed at 12 steps, most at 14.2 s (surprise 20190; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=139, end=144, force_N=[3.7, -0.2, -91.2], offset_mm=[0.6, 0.4, 0.3]).
  Ground truth: camera_shift at 4.8 s (offset=[0.05020010194954877, -0.00607423802277426, 0.003768934611418801]).

Episode 43: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 14.2 s, at_object at 10.9 s, grasped at 11.0 s.
  Surprise: my predictions failed at 4 steps, most at 5.9 s (surprise 3542; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=59, end=62, force_N=[54.7, -5.6, -133.6], offset_mm=[1.7, -1.7, 2.8]).
  Ground truth: camera_shift at 3.5 s (offset=[-0.02609050514898939, 0.021883446008644107, 0.004429766803881635]).

Episode 44: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 1.5 s, at_object at 4.6 s, grasped at 4.7 s.
  Surprise: my predictions failed at 1 steps, most at 2.5 s (surprise 693; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=25, camera_offset_mm=[-38.7, -6.4, 7.7], seen_jump=[0.0387, 0.0064, -0.0077]).
  Ground truth: camera_shift at 2.6 s (offset=[-0.03880953313809246, -0.006235425633004364, 0.007789756686980004]).

Episode 45: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 13.5 s, at_object at 8.8 s, grasped at 8.9 s.
  Surprise: my predictions failed at 1 steps, most at 2.8 s (surprise 702; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=28, camera_offset_mm=[37.4, -16.3, 1.4], seen_jump=[-0.0374, 0.0163, -0.0014]).
  Ground truth: camera_shift at 2.8 s (offset=[0.037285787978081714, -0.01640173867122438, 0.0014305966145952195]).

Episode 46: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 3.9 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s.
  Surprise: my predictions failed at 3 steps, most at 2.9 s (surprise 921; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=29, camera_offset_mm=[-20.7, 42.8, -1.8], seen_jump=[0.0207, -0.0428, 0.0018]).
  Ground truth: camera_shift at 2.9 s (offset=[-0.020871501352097425, 0.04303479707137788, -0.0021676199894367747]).

Episode 47: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 10.7 s, at_object at 14.7 s, grasped at 14.8 s.
  Surprise: my predictions failed at 1 steps, most at 2.2 s (surprise 592; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=22, camera_offset_mm=[28.4, -23.4, 2.5], seen_jump=[-0.0284, 0.0234, -0.0025]).
  Ground truth: camera_shift at 2.2 s (offset=[0.02840664495279875, -0.023417657994675833, 0.0024637428937208487]).

Episode 48: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.4 s, at_object at 11.1 s, grasped at 11.2 s.
  Surprise: my predictions failed at 1 steps, most at 2.4 s (surprise 1355; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=24, camera_offset_mm=[47.5, 27.7, -5.2], seen_jump=[-0.0475, -0.0277, 0.0052]).
  Ground truth: camera_shift at 2.5 s (offset=[0.047495256227065435, 0.027693442376505412, -0.005212611140140957]).

Episode 49: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.5 s, at_object at 11.1 s, grasped at 11.5 s, lifted at 3.0 s, over_target at 4.1 s.
  Surprise: my predictions failed at 43 steps, most at 10.7 s (surprise 6430; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=107, end=108, force_N=[7.9, -1.6, -38.5], offset_mm=[1.5, -0.5, 2.1]).
  Ground truth: camera_shift at 4.1 s (offset=[0.022664053881632354, -0.02224523134360853, -0.0032776587890867926]).

Episode 50: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 5.6 s, at_object at 2.9 s, grasped at 3.0 s, lifted at 3.7 s.
  Surprise: my predictions failed at 17 steps, most at 4.5 s (surprise 823; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=45, camera_offset_mm=[25.8, 35.4, -5.2], seen_jump=[-0.0258, -0.0354, 0.0052]).
  Ground truth: camera_shift at 4.5 s (offset=[0.025512793316172202, 0.035245329163077314, -0.0053871558201250514]).

Episode 51: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 14.2 s, at_object at 11.1 s, grasped at 11.2 s.
  Surprise: my predictions failed at 1 steps, most at 4.1 s (surprise 771; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=41, camera_offset_mm=[40.1, 13.6, -6.0], seen_jump=[-0.0401, -0.0136, 0.006]).
  Ground truth: camera_shift at 4.1 s (offset=[0.03990564968592261, 0.013528799163852663, -0.006029739109814893]).

Episode 52: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 2.3 s, at_object at 8.9 s, grasped at 9.0 s.
  Surprise: my predictions failed at 1 steps, most at 4.5 s (surprise 881; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=45, camera_offset_mm=[40.3, 25.5, 3.5], seen_jump=[-0.0403, -0.0255, -0.0035]).
  Ground truth: camera_shift at 4.6 s (offset=[0.03990891786734623, 0.02559264894137407, 0.003439897559127188]).

Episode 53: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.4 s.
  Surprise: my predictions failed at 1 steps, most at 2.0 s (surprise 1408; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=20, camera_offset_mm=[18.2, 55.3, -2.7], seen_jump=[-0.0182, -0.0553, 0.0027]).
  Ground truth: camera_shift at 2.0 s (offset=[0.01817299920991625, 0.05535670781682636, -0.0026977966351034288]).

Episode 54: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 5 time(s) and went back to re-grasp it.
  Progress: above_object at 14.1 s, at_object at 13.0 s, grasped at 13.1 s, lifted at 2.8 s, over_target at 4.0 s.
  Surprise: my predictions failed at 1 steps, most at 4.0 s (surprise 1051; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=40, camera_offset_mm=[38.5, 29.8, -1.2], seen_jump=[-0.0385, -0.0298, 0.0012]).
  Ground truth: camera_shift at 4.0 s (offset=[0.0385240351074759, 0.03007478597612591, -0.0011924569056843207]).

Episode 55: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s.
  Surprise: my predictions failed at 3 steps, most at 4.6 s (surprise 669; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=46, camera_offset_mm=[43.0, -12.6, -1.9], seen_jump=[-0.043, 0.0126, 0.0019]).
  Ground truth: camera_shift at 4.7 s (offset=[0.04317777058201739, -0.012664866137967249, -0.0014954275030184885]).

Episode 56: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 14.7 s, at_object at 13.1 s, grasped at 13.2 s.
  Surprise: my predictions failed at 1 steps, most at 3.3 s (surprise 1547; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=33, camera_offset_mm=[-43.5, -41.0, -0.8], seen_jump=[0.0435, 0.041, 0.0008]).
  Ground truth: camera_shift at 3.3 s (offset=[-0.043575893488745084, -0.04103060522755577, -0.0007990972138180782]).

Episode 57: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 5.6 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.6 s, over_target at 4.1 s.
  Surprise: my predictions failed at 51 steps, most at 7.9 s (surprise 15983; spikes start at 21).
  My explanation: something pushed my arm, probability 1.00 (onset=64, end=83, force_N=[-2.4, -0.1, -63.7], offset_mm=[-0.0, 0.0, 1.0]).
  Ground truth: camera_shift at 4.8 s (offset=[0.00218066744654777, -0.04486972192417547, 0.0005862432039354087]).

Episode 58: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s.
  Surprise: my predictions failed at 1 steps, most at 1.9 s (surprise 782; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=19, camera_offset_mm=[9.4, -41.5, 4.2], seen_jump=[-0.0094, 0.0415, -0.0042]).
  Ground truth: camera_shift at 2.0 s (offset=[0.009462281081844956, -0.041371382520200506, 0.004222857559794997]).

Episode 59: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 13.4 s, at_object at 5.8 s, grasped at 5.9 s, lifted at 3.4 s.
  Surprise: my predictions failed at 1 steps, most at 3.8 s (surprise 483; spikes start at 21).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=38, camera_offset_mm=[30.6, -13.8, 4.6], seen_jump=[-0.0306, 0.0138, -0.0046]).
  Ground truth: camera_shift at 3.9 s (offset=[0.030446401068337275, -0.013848608632825719, 0.004580302341526187]).
```
