# Episode reports (the agent's own account vs ground truth)

```
Episode 0: goal on(red, plate): SUCCESS after 124 planning steps (12.4 s).
  Progress: above_object at 1.0 s, at_object at 6.9 s, grasped at 7.0 s, lifted at 7.9 s, over_target at 10.1 s, lowered at 11.5 s, opened at 11.7 s, released at 12.4 s.
  Surprise: my predictions failed at 4 steps, most at 1.0 s (surprise 144; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=10, camera_offset_mm=[39.5, 25.0, 0.4], seen_jump=[-0.0395, -0.025, -0.0004]).
  Ground truth: no disturbance.

Episode 1: goal on(red, plate): SUCCESS after 141 planning steps (14.1 s).
  Progress: above_object at 1.3 s, at_object at 9.0 s, grasped at 9.1 s, lifted at 9.8 s, over_target at 12.0 s, lowered at 13.2 s, opened at 13.4 s, released at 14.1 s.
  Surprise: my predictions failed at 6 steps, most at 2.3 s (surprise 92; spikes start at 50).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 2: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.0 s, at_object at 3.0 s, grasped at 3.1 s, lifted at 4.0 s, over_target at 5.4 s, lowered at 6.7 s, opened at 6.9 s, released at 7.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 3: goal on(red, plate): SUCCESS after 107 planning steps (10.7 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 3.1 s, over_target at 4.7 s, lowered at 9.8 s, opened at 10.0 s, released at 10.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 4: goal on(red, plate): SUCCESS after 92 planning steps (9.2 s).
  Progress: above_object at 1.5 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 5.0 s, over_target at 6.9 s, lowered at 8.3 s, opened at 8.5 s, released at 9.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 5: goal on(red, plate): SUCCESS after 119 planning steps (11.9 s).
  Progress: above_object at 1.4 s, at_object at 7.4 s, grasped at 7.5 s, lifted at 8.6 s, over_target at 9.7 s, lowered at 11.0 s, opened at 11.2 s, released at 11.9 s.
  Surprise: my predictions failed at 2 steps, most at 10.7 s (surprise 180; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.65).
  Ground truth: no disturbance.

Episode 6: goal on(red, plate): SUCCESS after 66 planning steps (6.6 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s, over_target at 4.3 s, lowered at 5.7 s, opened at 5.9 s, released at 6.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 7: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 7.4 s, at_object at 9.8 s, grasped at 9.9 s, lifted at 10.9 s, over_target at 13.1 s, lowered at 14.8 s.
  Surprise: my predictions failed at 8 steps, most at 5.6 s (surprise 811; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=56, sink_mm=-0.0, slide_mm=[-4.0, 107.5], weight_lost_N=-0.0).
  What I changed: at 5.6 s, carry at 50% speed.
  Ground truth: no disturbance.

Episode 8: goal on(red, plate): SUCCESS after 55 planning steps (5.5 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.6 s, over_target at 3.5 s, lowered at 4.7 s, opened at 4.9 s, released at 5.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 9: goal on(red, plate): SUCCESS after 86 planning steps (8.6 s).
  Progress: above_object at 1.2 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.9 s, over_target at 4.0 s, lowered at 7.6 s, opened at 7.8 s, released at 8.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 10: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s, over_target at 3.9 s, lowered at 5.4 s, opened at 5.6 s, released at 6.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 11: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.5 s, over_target at 3.6 s, lowered at 5.5 s, opened at 5.8 s, released at 6.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 12: goal on(red, plate): SUCCESS after 75 planning steps (7.5 s).
  Progress: above_object at 1.6 s, at_object at 2.3 s, grasped at 2.4 s, lifted at 3.2 s, over_target at 5.3 s, lowered at 6.8 s, opened at 7.1 s, released at 7.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 13: goal on(red, plate): SUCCESS after 82 planning steps (8.2 s).
  Progress: above_object at 1.2 s, at_object at 3.5 s, grasped at 3.6 s, lifted at 4.4 s, over_target at 6.2 s, lowered at 7.5 s, opened at 7.8 s, released at 8.2 s.
  Surprise: my predictions failed at 3 steps, most at 6.4 s (surprise 335; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.75).
  Ground truth: no disturbance.

Episode 14: goal on(red, plate): SUCCESS after 124 planning steps (12.4 s).
  Progress: above_object at 1.0 s, at_object at 8.1 s, grasped at 8.2 s, lifted at 9.0 s, over_target at 10.2 s, lowered at 11.6 s, opened at 11.8 s, released at 12.4 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 15: goal on(red, plate): SUCCESS after 144 planning steps (14.4 s).
  Progress: above_object at 1.0 s, at_object at 9.9 s, grasped at 10.0 s, lifted at 10.7 s, over_target at 11.8 s, lowered at 13.4 s, opened at 13.6 s, released at 14.4 s.
  Surprise: my predictions failed at 2 steps, most at 14.0 s (surprise 66; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=140, camera_offset_mm=[-18.4, -4.2, -1.4], seen_jump=[0.0184, 0.0042, 0.0014]).
  Ground truth: no disturbance.

Episode 16: goal on(red, plate): SUCCESS after 83 planning steps (8.3 s).
  Progress: above_object at 1.1 s, at_object at 1.8 s, grasped at 2.3 s, lifted at 3.1 s, over_target at 5.8 s, lowered at 7.5 s, opened at 7.7 s, released at 8.3 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 17: goal on(red, plate): SUCCESS after 105 planning steps (10.5 s).
  Progress: above_object at 1.7 s, at_object at 5.9 s, grasped at 6.0 s, lifted at 6.9 s, over_target at 8.3 s, lowered at 9.7 s, opened at 9.9 s, released at 10.5 s.
  Surprise: my predictions failed at 1 steps, most at 8.7 s (surprise 55; spikes start at 50).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 18: goal on(red, plate): SUCCESS after 95 planning steps (9.5 s).
  Progress: above_object at 1.1 s, at_object at 1.9 s, grasped at 3.3 s, lifted at 4.7 s, over_target at 6.6 s, lowered at 8.6 s, opened at 8.8 s, released at 9.5 s.
  Surprise: my predictions failed at 4 steps, most at 1.4 s (surprise 161; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=14, camera_offset_mm=[33.8, 45.9, 0.8], seen_jump=[-0.0338, -0.0459, -0.0008]).
  Ground truth: no disturbance.

Episode 19: goal on(red, plate): SUCCESS after 75 planning steps (7.5 s).
  Progress: above_object at 1.0 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.3 s, over_target at 4.8 s, lowered at 6.5 s, opened at 6.7 s, released at 7.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 20: goal on(red, plate): SUCCESS after 67 planning steps (6.7 s).
  Progress: above_object at 0.9 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.5 s, over_target at 3.5 s, lowered at 5.7 s, opened at 5.9 s, released at 6.7 s.
  Surprise: my predictions failed at 7 steps, most at 3.6 s (surprise 7003; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=36, end=38, force_N=[23.1, 26.9, 0.1], offset_mm=[-0.6, -2.6, -1.8]).
  Ground truth: push at 3.6 s (force=[-23.078594408991457, -26.837222005850677, 0.0]).

Episode 21: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.9 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 3.8 s, over_target at 5.7 s, lowered at 6.8 s, opened at 7.1 s, released at 7.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: push at 2.6 s (force=[46.01616507284769, 4.795855691772432, 0.0]).

Episode 22: goal on(red, plate): SUCCESS after 84 planning steps (8.4 s).
  Progress: above_object at 1.0 s, at_object at 3.9 s, grasped at 4.0 s, lifted at 4.7 s, over_target at 5.8 s, lowered at 7.5 s, opened at 7.7 s, released at 8.4 s.
  Surprise: my predictions failed at 3 steps, most at 6.3 s (surprise 9885; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=62, end=64, force_N=[-35.7, 21.8, 0.1], offset_mm=[1.6, -0.3, 1.6]).
  Ground truth: push at 6.3 s (force=[35.959350551011255, -21.956566777307458, 0.0]).

Episode 23: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.0 s, at_object at 11.0 s, grasped at 11.1 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 1075; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=53, end=55, force_N=[47.0, 13.2, 0.1], offset_mm=[-3.1, -0.7, -1.8]).
  Ground truth: push at 5.3 s (force=[-46.88333296227451, -13.182722555874374, 0.0]).

Episode 24: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 9.1 s, at_object at 7.1 s, grasped at 7.2 s.
  Surprise: my predictions failed at 7 steps, most at 7.4 s (surprise 4022; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=74, sink_mm=3.9, slide_mm=[-118.5, -75.5], weight_lost_N=-0.0).
  What I changed: at 5.0 s, carry at 50% speed.
  Ground truth: push at 4.0 s (force=[12.083944329303097, -27.518493588925867, 0.0]).

Episode 25: goal on(red, plate): SUCCESS after 114 planning steps (11.4 s).
  Progress: above_object at 1.2 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 6.3 s, over_target at 8.9 s, lowered at 10.6 s, opened at 10.8 s, released at 11.4 s.
  Surprise: my predictions failed at 10 steps, most at 6.0 s (surprise 10965; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=58, end=60, force_N=[-43.7, -9.4, 0.0], offset_mm=[1.8, -0.8, 1.9]).
  Ground truth: push at 5.8 s (force=[43.60389360190877, 9.340547450299931, 0.0]).

Episode 26: goal on(red, plate): SUCCESS after 91 planning steps (9.1 s).
  Progress: above_object at 2.9 s, at_object at 4.4 s, grasped at 4.5 s, lifted at 5.2 s, over_target at 6.6 s, lowered at 8.2 s, opened at 8.4 s, released at 9.1 s.
  Surprise: my predictions failed at 3 steps, most at 2.5 s (surprise 12480; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=24, end=26, force_N=[-21.2, -41.7, 0.2], offset_mm=[0.5, 1.9, 0.4]).
  Ground truth: push at 2.5 s (force=[21.28355781425198, 42.20018974344667, 0.0]).

Episode 27: goal on(red, plate): SUCCESS after 82 planning steps (8.2 s).
  Progress: above_object at 2.2 s, at_object at 2.9 s, grasped at 3.0 s, lifted at 4.0 s, over_target at 6.4 s, lowered at 7.5 s, opened at 7.8 s, released at 8.2 s.
  Surprise: my predictions failed at 5 steps, most at 4.7 s (surprise 8132; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=45, end=47, force_N=[11.8, -36.5, 0.0], offset_mm=[-1.0, 2.2, -0.0]).
  Ground truth: push at 4.6 s (force=[-11.816637376833446, 36.593135166683, 0.0]).

Episode 28: goal on(red, plate): SUCCESS after 83 planning steps (8.3 s).
  Progress: above_object at 2.2 s, at_object at 3.8 s, grasped at 3.9 s, lifted at 5.2 s, over_target at 6.3 s, lowered at 7.3 s, opened at 7.5 s, released at 8.3 s.
  Surprise: my predictions failed at 3 steps, most at 2.4 s (surprise 225; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=24, end=26, force_N=[-32.0, -5.7, -0.0], offset_mm=[1.7, -0.0, 1.0]).
  Ground truth: push at 2.4 s (force=[31.972743043725643, 5.749971059610389, 0.0]).

Episode 29: goal on(red, plate): SUCCESS after 57 planning steps (5.7 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s, over_target at 3.8 s, lowered at 5.0 s, opened at 5.3 s, released at 5.7 s.
  Surprise: my predictions failed at 3 steps, most at 5.1 s (surprise 1851; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=50, end=52, force_N=[25.0, 32.6, -0.1], offset_mm=[-1.4, -3.7, -0.4]).
  Ground truth: push at 5.1 s (force=[-25.468354696480795, -33.783199626513586, 0.0]).

Episode 30: goal on(red, plate): SUCCESS after 104 planning steps (10.4 s).
  Progress: above_object at 1.1 s, at_object at 5.4 s, grasped at 5.5 s, lifted at 7.0 s, over_target at 8.4 s, lowered at 9.5 s, opened at 9.7 s, released at 10.4 s.
  Surprise: my predictions failed at 3 steps, most at 3.6 s (surprise 107; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=40, end=40, force_N=[-0.7, -3.6, 1.7], offset_mm=[-0.1, 0.0, -1.1]).
  Ground truth: push at 5.6 s (force=[-37.1879651314092, 33.338839733062635, 0.0]).

Episode 31: goal on(red, plate): SUCCESS after 115 planning steps (11.5 s).
  Progress: above_object at 1.0 s, at_object at 7.2 s, grasped at 7.3 s, lifted at 8.0 s, over_target at 8.8 s, lowered at 10.8 s, opened at 11.1 s, released at 11.5 s.
  Surprise: my predictions failed at 1 steps, most at 6.3 s (surprise 76; spikes start at 50).
  My explanation: nothing unusual happened.
  Ground truth: push at 6.3 s (force=[-16.946491428273678, -39.529817452474624, 0.0]).

Episode 32: goal on(red, plate): SUCCESS after 93 planning steps (9.3 s).
  Progress: above_object at 1.0 s, at_object at 4.4 s, grasped at 4.5 s, lifted at 5.6 s, over_target at 6.7 s, lowered at 8.5 s, opened at 8.7 s, released at 9.3 s.
  Surprise: my predictions failed at 3 steps, most at 5.9 s (surprise 7985; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=59, end=61, force_N=[14.3, 35.1, 0.1], offset_mm=[-1.6, -2.5, -1.3]).
  Ground truth: push at 5.9 s (force=[-14.249305886181846, -34.988097121005616, 0.0]).

Episode 33: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.3 s, at_object at 3.3 s.
  Surprise: my predictions failed at 1 steps, most at 2.9 s (surprise 145; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=29, end=31, force_N=[7.1, 40.0, 3.9], offset_mm=[-1.0, -4.4, -0.5]).
  Ground truth: push at 2.9 s (force=[-7.217851567027591, -39.85883429535791, 0.0]).

Episode 34: goal on(red, plate): SUCCESS after 72 planning steps (7.2 s).
  Progress: above_object at 1.5 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.2 s, over_target at 4.4 s, lowered at 6.4 s, opened at 6.6 s, released at 7.2 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 8598; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[14.7, -36.7, -0.0], offset_mm=[-0.1, 3.4, -0.1]).
  Ground truth: push at 5.3 s (force=[-14.676798931125873, 36.90539800747222, 0.0]).

Episode 35: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 0.9 s, at_object at 1.6 s, grasped at 2.5 s, lifted at 3.4 s, over_target at 5.2 s, lowered at 6.6 s, opened at 6.8 s, released at 7.3 s.
  Surprise: my predictions failed at 4 steps, most at 5.9 s (surprise 7654; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=59, end=61, force_N=[-34.1, 15.0, -0.0], offset_mm=[1.3, -0.1, 1.4]).
  Ground truth: push at 5.9 s (force=[34.01070499148584, -14.961054212462347, 0.0]).

Episode 36: goal on(red, plate): SUCCESS after 90 planning steps (9.0 s).
  Progress: above_object at 4.2 s, at_object at 5.1 s, grasped at 5.2 s, lifted at 5.9 s, over_target at 7.2 s, lowered at 8.3 s, opened at 8.6 s, released at 9.0 s.
  Surprise: my predictions failed at 3 steps, most at 5.4 s (surprise 6116; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[32.2, 16.0, -0.7], offset_mm=[-1.0, -0.1, 0.1]).
  Ground truth: push at 5.3 s (force=[-32.81888686057621, -15.830473665954154, 0.0]).

Episode 37: goal on(red, plate): SUCCESS after 122 planning steps (12.2 s).
  Progress: above_object at 1.1 s, at_object at 7.6 s, grasped at 7.7 s, lifted at 8.7 s, over_target at 10.0 s, lowered at 11.4 s, opened at 11.6 s, released at 12.2 s.
  Surprise: my predictions failed at 1 steps, most at 10.9 s (surprise 50; spikes start at 50).
  My explanation: nothing unusual happened.
  Ground truth: push at 4.8 s (force=[-19.850645061690944, 32.20622405527275, 0.0]).

Episode 38: goal on(red, plate): SUCCESS after 92 planning steps (9.2 s).
  Progress: above_object at 1.1 s, at_object at 3.4 s, grasped at 3.6 s, lifted at 4.9 s, over_target at 6.4 s, lowered at 8.2 s, opened at 8.4 s, released at 9.2 s.
  Surprise: my predictions failed at 7 steps, most at 5.2 s (surprise 6604; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[-26.7, 22.0, -0.0], offset_mm=[0.3, -0.8, -0.1]).
  Ground truth: push at 5.2 s (force=[26.65387221960392, -21.972720288144398, 0.0]).

Episode 39: goal on(red, plate): SUCCESS after 88 planning steps (8.8 s).
  Progress: above_object at 1.0 s, at_object at 3.3 s, grasped at 3.4 s, lifted at 4.1 s, over_target at 6.2 s, lowered at 7.9 s, opened at 8.1 s, released at 8.8 s.
  Surprise: my predictions failed at 3 steps, most at 4.9 s (surprise 11893; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=48, end=50, force_N=[-40.3, -23.2, 0.4], offset_mm=[1.0, 2.2, 2.1]).
  Ground truth: push at 4.9 s (force=[40.3022527572971, 23.49935978956869, 0.0]).

Episode 40: goal on(red, plate): SUCCESS after 60 planning steps (6.0 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s, over_target at 3.7 s, lowered at 5.2 s, opened at 5.4 s, released at 6.0 s.
  Surprise: my predictions failed at 15 steps, most at 2.8 s (surprise 163; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.27, extra_mass_kg=0.537, sag_mm=0.28).
  What I changed: at 3.3 s, expect the extra weight (extra_mass_kg=0.526).
  Ground truth: payload at 0.0 s (mass=0.536129492246605).

Episode 41: goal on(red, plate): SUCCESS after 78 planning steps (7.8 s).
  Progress: above_object at 1.5 s, at_object at 2.3 s, grasped at 2.4 s, lifted at 3.3 s, over_target at 5.4 s, lowered at 6.6 s, opened at 6.8 s, released at 7.8 s.
  Surprise: my predictions failed at 1 steps, most at 2.7 s (surprise 74; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.58, extra_mass_kg=0.365, sag_mm=0.2).
  What I changed: at 2.7 s, expect the extra weight (extra_mass_kg=0.37).
  What I changed: at 3.0 s, expect the extra weight (extra_mass_kg=0.37).
  Ground truth: payload at 0.0 s (mass=0.37181083289788563).

Episode 42: goal on(red, plate): SUCCESS after 66 planning steps (6.6 s).
  Progress: above_object at 1.0 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.8 s, over_target at 4.3 s, lowered at 5.8 s, opened at 6.0 s, released at 6.6 s.
  Surprise: my predictions failed at 1 steps, most at 2.3 s (surprise 154; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.54, extra_mass_kg=0.565, sag_mm=0.27).
  What I changed: at 2.3 s, expect the extra weight (extra_mass_kg=0.564).
  What I changed: at 2.6 s, expect the extra weight (extra_mass_kg=0.55).
  Ground truth: payload at 0.0 s (mass=0.5629452692432111).

Episode 43: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.0 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.6 s, over_target at 5.2 s, lowered at 6.4 s, opened at 6.6 s, released at 7.3 s.
  Surprise: my predictions failed at 3 steps, most at 6.3 s (surprise 93; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.15, extra_mass_kg=0.321, sag_mm=0.11).
  What I changed: at 3.1 s, expect the extra weight (extra_mass_kg=0.316).
  What I changed: at 3.4 s, expect the extra weight (extra_mass_kg=0.312).
  Ground truth: payload at 0.0 s (mass=0.3175704104415583).

Episode 44: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 13.9 s, at_object at 14.7 s, grasped at 14.8 s.
  Surprise: my predictions failed at 7 steps, most at 12.8 s (surprise 3480; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=128, sink_mm=2.8, slide_mm=[-129.8, -6.6], weight_lost_N=-0.0).
  What I changed: at 6.0 s, carry at 50% speed.
  Ground truth: payload at 0.0 s (mass=0.4008351181636981).

Episode 45: goal on(red, plate): SUCCESS after 70 planning steps (7.0 s).
  Progress: above_object at 1.5 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.6 s, over_target at 4.8 s, lowered at 6.2 s, opened at 6.4 s, released at 7.0 s.
  Surprise: my predictions failed at 16 steps, most at 2.3 s (surprise 161; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.42, extra_mass_kg=0.349, sag_mm=0.0).
  What I changed: at 3.9 s, expect the extra weight (extra_mass_kg=0.341).
  What I changed: at 4.2 s, expect the extra weight (extra_mass_kg=0.342).
  What I changed: at 5.9 s, expect the extra weight (extra_mass_kg=0.348).
  What I changed: at 6.2 s, expect the extra weight (extra_mass_kg=0.349).
  Ground truth: payload at 0.0 s (mass=0.3450838400684517).

Episode 46: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.0 s, at_object at 2.9 s, grasped at 3.0 s, lifted at 3.9 s, over_target at 5.4 s, lowered at 6.7 s, opened at 6.9 s, released at 7.3 s.
  Surprise: my predictions failed at 1 steps, most at 3.4 s (surprise 101; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.31, extra_mass_kg=0.439, sag_mm=0.08).
  What I changed: at 3.4 s, expect the extra weight (extra_mass_kg=0.439).
  What I changed: at 3.7 s, expect the extra weight (extra_mass_kg=0.425).
  Ground truth: payload at 0.0 s (mass=0.4351018099947861).

Episode 47: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 4 time(s) and went back to re-grasp it.
  Progress: above_object at 12.4 s, at_object at 14.3 s, grasped at 14.4 s.
  Surprise: my predictions failed at 5 steps, most at 3.1 s (surprise 150; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=29, end=31, force_N=[-2.3, -0.7, 2.8], offset_mm=[0.3, 0.5, 1.0]).
  Ground truth: payload at 0.0 s (mass=0.5388972810861883).

Episode 48: goal on(red, plate): SUCCESS after 55 planning steps (5.5 s).
  Progress: above_object at 1.4 s, at_object at 2.2 s, grasped at 2.3 s, lifted at 3.0 s, over_target at 3.9 s, lowered at 4.9 s, opened at 5.1 s, released at 5.5 s.
  Surprise: my predictions failed at 2 steps, most at 2.5 s (surprise 73; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.64, extra_mass_kg=0.371, sag_mm=0.36).
  What I changed: at 2.5 s, expect the extra weight (extra_mass_kg=0.374).
  What I changed: at 2.8 s, expect the extra weight (extra_mass_kg=0.365).
  Ground truth: payload at 0.0 s (mass=0.36919266269812423).

Episode 49: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 3.1 s, over_target at 4.3 s, lowered at 5.8 s, opened at 6.0 s, released at 6.4 s.
  Surprise: my predictions failed at 2 steps, most at 2.5 s (surprise 53; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.13, extra_mass_kg=0.319, sag_mm=0.33).
  What I changed: at 2.6 s, expect the extra weight (extra_mass_kg=0.317).
  What I changed: at 2.9 s, expect the extra weight (extra_mass_kg=0.309).
  Ground truth: payload at 0.0 s (mass=0.31560639031932286).

Episode 50: goal on(red, plate): SUCCESS after 144 planning steps (14.4 s).
  Progress: above_object at 1.0 s, at_object at 9.9 s, grasped at 10.0 s, lifted at 10.7 s, over_target at 12.0 s, lowered at 13.6 s, opened at 13.9 s, released at 14.4 s.
  Surprise: my predictions failed at 6 steps, most at 13.4 s (surprise 474; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=119, end=130, force_N=[-0.1, -0.1, 4.2], offset_mm=[-0.2, -0.3, 0.0]).
  What I changed: at 10.1 s, expect the extra weight (extra_mass_kg=0.418).
  What I changed: at 10.4 s, expect the extra weight (extra_mass_kg=0.41).
  Ground truth: payload at 0.0 s (mass=0.4213655519464584).

Episode 51: goal on(red, plate): SUCCESS after 69 planning steps (6.9 s).
  Progress: above_object at 1.7 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 4.1 s, over_target at 5.0 s, lowered at 6.2 s, opened at 6.5 s, released at 6.9 s.
  Surprise: my predictions failed at 1 steps, most at 3.5 s (surprise 60; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.53, extra_mass_kg=0.36, sag_mm=0.42).
  What I changed: at 3.5 s, expect the extra weight (extra_mass_kg=0.353).
  What I changed: at 3.8 s, expect the extra weight (extra_mass_kg=0.35).
  Ground truth: payload at 0.0 s (mass=0.3595539133527766).

Episode 52: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.1 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 3.1 s, over_target at 4.2 s, lowered at 6.3 s, opened at 6.5 s, released at 7.3 s.
  Surprise: my predictions held throughout.
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.22, extra_mass_kg=0.329, sag_mm=0.26).
  What I changed: at 2.7 s, expect the extra weight (extra_mass_kg=0.321).
  What I changed: at 3.0 s, expect the extra weight (extra_mass_kg=0.32).
  Ground truth: payload at 0.0 s (mass=0.32722591368573656).

Episode 53: goal on(red, plate): SUCCESS after 109 planning steps (10.9 s).
  Progress: above_object at 1.6 s, at_object at 5.3 s, grasped at 5.4 s, lifted at 6.6 s, over_target at 8.5 s, lowered at 10.1 s, opened at 10.3 s, released at 10.9 s.
  Surprise: my predictions failed at 6 steps, most at 10.7 s (surprise 248; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=1.51).
  What I changed: at 6.1 s, expect the extra weight (extra_mass_kg=0.48).
  What I changed: at 6.4 s, expect the extra weight (extra_mass_kg=0.472).
  What I changed: at 8.0 s, expect the extra weight (extra_mass_kg=0.473).
  What I changed: at 8.3 s, expect the extra weight (extra_mass_kg=0.473).
  What I changed: at 10.0 s, expect the extra weight (extra_mass_kg=0.478).
  What I changed: at 10.3 s, expect the extra weight (extra_mass_kg=0.48).
  Ground truth: payload at 0.0 s (mass=0.47409971579605514).

Episode 54: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 0.9 s, at_object at 1.5 s, grasped at 1.8 s, lifted at 2.8 s, over_target at 3.9 s, lowered at 5.6 s, opened at 5.8 s, released at 6.4 s.
  Surprise: my predictions failed at 2 steps, most at 2.3 s (surprise 80; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.85, extra_mass_kg=0.392, sag_mm=0.23).
  What I changed: at 2.3 s, expect the extra weight (extra_mass_kg=0.386).
  What I changed: at 2.6 s, expect the extra weight (extra_mass_kg=0.382).
  Ground truth: payload at 0.0 s (mass=0.38960883984567674).

Episode 55: goal on(red, plate): SUCCESS after 119 planning steps (11.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 7.2 s, at_object at 8.0 s, grasped at 8.1 s, lifted at 9.1 s, over_target at 9.7 s, lowered at 11.1 s, opened at 11.3 s, released at 11.9 s.
  Surprise: my predictions failed at 4 steps, most at 6.4 s (surprise 303; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.94, extra_mass_kg=0.504, sag_mm=0.23).
  What I changed: at 3.3 s, expect the extra weight (extra_mass_kg=0.499).
  What I changed: at 3.6 s, expect the extra weight (extra_mass_kg=0.497).
  What I changed: at 11.8 s, expect the extra weight (extra_mass_kg=0.512).
  Ground truth: payload at 0.0 s (mass=0.5015984633869078).

Episode 56: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 10.5 s, at_object at 14.0 s, grasped at 14.1 s.
  Surprise: my predictions failed at 2 steps, most at 14.2 s (surprise 130; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=140, end=142, force_N=[-2.3, -2.8, 0.2], offset_mm=[-0.6, -0.4, 0.2]).
  Ground truth: payload at 0.0 s (mass=0.35985463319046396).

Episode 57: goal on(red, plate): SUCCESS after 61 planning steps (6.1 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.9 s, over_target at 4.2 s, lowered at 5.4 s, opened at 5.6 s, released at 6.1 s.
  Surprise: my predictions failed at 4 steps, most at 2.1 s (surprise 150; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.73, extra_mass_kg=0.584, sag_mm=0.27).
  What I changed: at 2.4 s, expect the extra weight (extra_mass_kg=0.605).
  What I changed: at 5.7 s, expect the extra weight (extra_mass_kg=0.588).
  What I changed: at 6.0 s, expect the extra weight (extra_mass_kg=0.588).
  Ground truth: payload at 0.0 s (mass=0.5826339331519493).

Episode 58: goal on(red, plate): SUCCESS after 97 planning steps (9.7 s).
  Progress: above_object at 1.5 s, at_object at 5.4 s, grasped at 5.5 s, lifted at 6.2 s, over_target at 7.3 s, lowered at 8.9 s, opened at 9.1 s, released at 9.7 s.
  Surprise: my predictions failed at 3 steps, most at 1.7 s (surprise 183; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=17, camera_offset_mm=[34.4, 48.9, 1.1], seen_jump=[-0.0344, -0.0489, -0.0011]).
  What I changed: at 5.6 s, expect the extra weight (extra_mass_kg=0.416).
  What I changed: at 5.9 s, expect the extra weight (extra_mass_kg=0.406).
  Ground truth: payload at 0.0 s (mass=0.40953305047344857).

Episode 59: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s, over_target at 4.1 s, lowered at 5.5 s, opened at 5.7 s, released at 6.2 s.
  Surprise: my predictions failed at 2 steps, most at 2.1 s (surprise 58; spikes start at 50).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.24, extra_mass_kg=0.33, sag_mm=0.09).
  What I changed: at 2.2 s, expect the extra weight (extra_mass_kg=0.328).
  What I changed: at 2.5 s, expect the extra weight (extra_mass_kg=0.323).
  Ground truth: payload at 0.0 s (mass=0.3316485838710688).

Episode 60: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 7.0 s, at_object at 12.3 s, grasped at 12.4 s, lifted at 13.6 s.
  Surprise: my predictions failed at 13 steps, most at 6.4 s (surprise 3485; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=64, sink_mm=139.5, slide_mm=[-17.9, 9.7], weight_lost_N=0.71).
  What I changed: at 6.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 61: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.6 s, at_object at 9.8 s, grasped at 10.1 s, lifted at 3.2 s.
  Surprise: my predictions failed at 6 steps, most at 10.3 s (surprise 1564; spikes start at 50).
  My explanation: something pushed my arm, probability 1.00 (onset=88, end=103, force_N=[-8.5, -2.7, -19.8], offset_mm=[-0.2, 0.0, 0.2]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 62: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 10.3 s, at_object at 8.0 s, grasped at 8.1 s, lifted at 2.7 s.
  Surprise: my predictions failed at 6 steps, most at 9.1 s (surprise 1549; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.77).
  What I changed: at 3.3 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 63: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.1 s.
  Surprise: my predictions failed at 4 steps, most at 11.3 s (surprise 68; spikes start at 50).
  My explanation: nothing unusual happened.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 64: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 6.2 s, at_object at 7.3 s, grasped at 7.4 s.
  Surprise: my predictions failed at 3 steps, most at 5.2 s (surprise 1242; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=52, sink_mm=-0.0, slide_mm=[-57.8, -56.7], weight_lost_N=-0.0).
  What I changed: at 2.8 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 65: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.6 s.
  Surprise: my predictions failed at 4 steps, most at 9.1 s (surprise 281; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.41).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 66: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 8.8 s, at_object at 13.7 s, grasped at 13.8 s, lifted at 6.7 s.
  Surprise: my predictions failed at 3 steps, most at 7.7 s (surprise 707; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=77, camera_offset_mm=[-65.9, 51.2, 0.8], seen_jump=[0.0659, -0.0512, -0.0008]).
  What I changed: at 7.5 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 67: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 1.1 s, at_object at 13.3 s, grasped at 13.4 s.
  Surprise: my predictions failed at 5 steps, most at 5.6 s (surprise 412; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.59).
  What I changed: at 13.7 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 68: goal on(red, plate): SUCCESS after 121 planning steps (12.1 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 6.3 s, at_object at 7.8 s, grasped at 7.9 s, lifted at 9.1 s, over_target at 9.6 s, lowered at 11.3 s, opened at 11.5 s, released at 12.1 s.
  Surprise: my predictions failed at 2 steps, most at 5.4 s (surprise 1293; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=54, sink_mm=100.2, slide_mm=[-8.4, -33.4], weight_lost_N=0.64).
  What I changed: at 5.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 69: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 5 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 13.7 s, grasped at 13.8 s, lifted at 3.0 s.
  Surprise: my predictions failed at 5 steps, most at 14.4 s (surprise 7705; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=144, sink_mm=-0.0, slide_mm=[-195.7, 67.5], weight_lost_N=-0.0).
  What I changed: at 3.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 70: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 4.8 s, at_object at 7.3 s, grasped at 7.4 s, lifted at 4.0 s.
  Surprise: my predictions failed at 30 steps, most at 7.9 s (surprise 13681; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=79, sink_mm=9.2, slide_mm=[-102.1, -245.2], weight_lost_N=-0.0).
  What I changed: at 7.9 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 71: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.4 s, at_object at 10.1 s, grasped at 10.2 s, lifted at 3.8 s.
  Surprise: my predictions failed at 6 steps, most at 11.2 s (surprise 31766; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=112, sink_mm=10.0, slide_mm=[-419.0, -220.0], weight_lost_N=-0.0).
  What I changed: at 11.2 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 72: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 9.3 s, at_object at 11.7 s, grasped at 11.9 s, lifted at 7.2 s, over_target at 8.6 s.
  Surprise: my predictions failed at 5 steps, most at 8.7 s (surprise 3865; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=87, sink_mm=168.3, slide_mm=[-20.1, -35.9], weight_lost_N=0.63).
  What I changed: at 8.7 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 73: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 12.3 s, at_object at 9.5 s, grasped at 9.6 s, lifted at 10.4 s.
  Surprise: my predictions failed at 6 steps, most at 12.0 s (surprise 109; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.76).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 74: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 5 time(s) and went back to re-grasp it.
  Progress: above_object at 14.4 s, at_object at 12.6 s, grasped at 12.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 75: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.4 s, at_object at 9.0 s, grasped at 9.1 s, lifted at 10.3 s, over_target at 11.0 s.
  Surprise: my predictions failed at 5 steps, most at 11.6 s (surprise 643; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=116, sink_mm=100.7, slide_mm=[-42.1, 11.8], weight_lost_N=0.6).
  What I changed: at 6.0 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 76: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 9.7 s, at_object at 10.9 s, grasped at 6.7 s, lifted at 7.7 s.
  Surprise: my predictions failed at 6 steps, most at 11.5 s (surprise 1130; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.74).
  What I changed: at 3.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 77: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 14.4 s, grasped at 14.5 s, lifted at 10.5 s, over_target at 12.0 s.
  Surprise: my predictions failed at 6 steps, most at 12.1 s (surprise 4128; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=121, sink_mm=158.1, slide_mm=[-10.9, 8.8], weight_lost_N=0.66).
  What I changed: at 12.1 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 78: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 4 time(s) and went back to re-grasp it.
  Progress: above_object at 13.1 s, at_object at 14.0 s, grasped at 14.1 s.
  Surprise: my predictions failed at 9 steps, most at 12.0 s (surprise 7277; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=120, sink_mm=6.5, slide_mm=[-92.3, -213.2], weight_lost_N=-0.0).
  What I changed: at 5.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 79: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 4.6 s, at_object at 6.8 s, grasped at 6.9 s, lifted at 2.8 s.
  Surprise: my predictions failed at 3 steps, most at 4.1 s (surprise 6107; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=41, sink_mm=159.4, slide_mm=[41.7, 56.9], weight_lost_N=0.62).
  What I changed: at 4.3 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 80: goal on(red, plate): SUCCESS after 55 planning steps (5.5 s).
  Progress: above_object at 0.9 s, at_object at 1.5 s, grasped at 1.6 s, lifted at 2.6 s, over_target at 3.6 s, lowered at 4.8 s, opened at 5.1 s, released at 5.5 s.
  Surprise: my predictions failed at 3 steps, most at 1.1 s (surprise 142; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=11, camera_offset_mm=[45.5, 19.2, -0.1], seen_jump=[-0.0455, -0.0192, 0.0001]).
  Ground truth: camera_shift at 2.5 s (offset=[-0.039812385659612884, -0.04192262101013534, 0.009091809873814744]).

Episode 81: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.4 s, at_object at 2.3 s, grasped at 2.4 s, lifted at 3.1 s, over_target at 5.2 s, lowered at 6.5 s, opened at 6.7 s, released at 7.3 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.6 s (offset=[-0.04275684958417764, 2.798957440961015e-05, 0.0024042690403075564]).

Episode 82: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.1 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.9 s, over_target at 4.3 s, lowered at 5.7 s, opened at 5.9 s, released at 6.4 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 0.9 s (offset=[0.058440562524408976, -0.001801098304528124, 0.005154576906165829]).

Episode 83: goal on(red, plate): SUCCESS after 72 planning steps (7.2 s).
  Progress: above_object at 1.1 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.2 s, over_target at 4.9 s, lowered at 6.3 s, opened at 6.5 s, released at 7.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.7 s (offset=[-0.04587334934213697, 0.0007429233657669993, 0.005715714014276148]).

Episode 84: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.1 s, at_object at 1.8 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 3.9 s (offset=[-0.044730906450852495, 0.02658450569052531, 0.008641193732267566]).

Episode 85: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.2 s.
  Surprise: my predictions failed at 17 steps, most at 13.9 s (surprise 301; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 0.83 (onset=139, camera_offset_mm=[1.6, 17.8, 2.2], seen_jump=[-0.0016, -0.0178, -0.0022]).
  Ground truth: camera_shift at 3.8 s (offset=[0.03892311737917909, 0.03428607415638614, 0.008548478572491197]).

Episode 86: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 12.2 s, grasped at 12.3 s, lifted at 13.0 s, over_target at 14.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.3 s (offset=[0.029825130379335377, -0.0060932417200100985, 0.009623900801326886]).

Episode 87: goal on(red, plate): SUCCESS after 116 planning steps (11.6 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 4.5 s, at_object at 5.3 s, grasped at 5.4 s, lifted at 6.6 s, over_target at 9.0 s, lowered at 10.7 s, opened at 10.9 s, released at 11.6 s.
  Surprise: my predictions failed at 7 steps, most at 3.2 s (surprise 14278; spikes start at 50).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=32, sink_mm=3.3, slide_mm=[-178.0, -265.4], weight_lost_N=-0.0).
  What I changed: at 3.2 s, carry at 50% speed.
  Ground truth: camera_shift at 4.4 s (offset=[0.03322485056448212, -0.009154353346265796, 0.009452576276459101]).

Episode 88: goal on(red, plate): SUCCESS after 112 planning steps (11.2 s).
  Progress: above_object at 1.0 s, at_object at 7.0 s, grasped at 7.1 s, lifted at 7.8 s, over_target at 9.0 s, lowered at 10.4 s, opened at 10.6 s, released at 11.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.8 s (offset=[0.04211078077803654, -0.034865798038288626, -0.005352541607213923]).

Episode 89: goal on(red, plate): SUCCESS after 84 planning steps (8.4 s).
  Progress: above_object at 1.0 s, at_object at 4.5 s, grasped at 4.6 s, lifted at 5.4 s, over_target at 6.4 s, lowered at 7.7 s, opened at 8.0 s, released at 8.4 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.8 s (offset=[0.01847932739603208, -0.05466704589629126, -0.004677394554154148]).

Episode 90: goal on(red, plate): SUCCESS after 59 planning steps (5.9 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.6 s, over_target at 3.8 s, lowered at 5.2 s, opened at 5.5 s, released at 5.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.4 s (offset=[-0.041993910668189416, -0.010483015775755547, -0.009189785776231307]).

Episode 91: goal on(red, plate): SUCCESS after 55 planning steps (5.5 s).
  Progress: above_object at 1.1 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.7 s, over_target at 3.6 s, lowered at 4.7 s, opened at 5.0 s, released at 5.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 4.7 s (offset=[-0.0054638965435530274, -0.048121998264614184, -0.009432692697729578]).

Episode 92: goal on(red, plate): SUCCESS after 88 planning steps (8.8 s).
  Progress: above_object at 2.7 s, at_object at 3.5 s, grasped at 3.6 s, lifted at 4.4 s, over_target at 5.6 s, lowered at 7.9 s, opened at 8.1 s, released at 8.8 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.2 s (offset=[-0.0058580413491001, -0.029911513277774877, 0.00025517446524156093]).

Episode 93: goal on(red, plate): SUCCESS after 84 planning steps (8.4 s).
  Progress: above_object at 2.0 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 3.9 s, over_target at 5.8 s, lowered at 7.5 s, opened at 7.7 s, released at 8.4 s.
  Surprise: my predictions failed at 4 steps, most at 8.3 s (surprise 219; spikes start at 50).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.82).
  Ground truth: camera_shift at 4.0 s (offset=[0.028861506665298697, -0.01378013548003836, 0.006826345592247663]).

Episode 94: goal on(red, plate): SUCCESS after 79 planning steps (7.9 s).
  Progress: above_object at 1.1 s, at_object at 2.2 s, grasped at 2.3 s, lifted at 3.0 s, over_target at 5.4 s, lowered at 7.0 s, opened at 7.2 s, released at 7.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.7 s (offset=[0.03684024207957708, 0.016408807108357116, 0.009321241615681404]).

Episode 95: goal on(red, plate): SUCCESS after 79 planning steps (7.9 s).
  Progress: above_object at 1.1 s, at_object at 3.5 s, grasped at 3.8 s, lifted at 4.6 s, over_target at 5.9 s, lowered at 7.2 s, opened at 7.5 s, released at 7.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.6 s (offset=[-0.03491547804767095, -0.014393590596098065, -0.005166485718113101]).

Episode 96: goal on(red, plate): SUCCESS after 109 planning steps (10.9 s).
  Progress: above_object at 1.8 s, at_object at 6.3 s, grasped at 6.4 s, lifted at 7.2 s, over_target at 8.5 s, lowered at 10.1 s, opened at 10.3 s, released at 10.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.3 s (offset=[0.028057332016747115, -0.02377533154505764, -0.004233384859848448]).

Episode 97: goal on(red, plate): SUCCESS after 71 planning steps (7.1 s).
  Progress: above_object at 1.5 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.2 s, over_target at 4.9 s, lowered at 6.3 s, opened at 6.5 s, released at 7.1 s.
  Surprise: my predictions failed at 4 steps, most at 6.6 s (surprise 83; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=66, camera_offset_mm=[32.9, 2.8, 0.1], seen_jump=[-0.0329, -0.0028, -0.0001]).
  Ground truth: camera_shift at 1.3 s (offset=[-0.03996166319853981, -0.024015474895515043, 0.006194215518255554]).

Episode 98: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 2.9 s, lifted at 3.7 s, over_target at 4.8 s, lowered at 6.5 s, opened at 6.7 s, released at 7.3 s.
  Surprise: my predictions failed at 2 steps, most at 1.5 s (surprise 169; spikes start at 50).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=15, camera_offset_mm=[-32.0, -55.7, -0.9], seen_jump=[0.032, 0.0557, 0.0009]).
  Ground truth: camera_shift at 2.7 s (offset=[-0.03589560013521975, -0.014336394042755926, 0.006362419419418208]).

Episode 99: goal on(red, plate): SUCCESS after 115 planning steps (11.5 s).
  Progress: above_object at 1.1 s, at_object at 4.4 s, grasped at 4.5 s, lifted at 7.7 s, over_target at 9.1 s, lowered at 10.6 s, opened at 10.8 s, released at 11.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.5 s (offset=[-0.04116309146536361, -0.04194981037048178, -0.0026119117781663826]).
```
