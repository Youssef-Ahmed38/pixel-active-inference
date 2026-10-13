# Episode reports (the agent's own account vs ground truth)

```
Episode 0: goal on(red, plate): SUCCESS after 57 planning steps (5.7 s).
  Progress: above_object at 1.1 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 3.0 s, over_target at 4.0 s, lowered at 5.0 s, opened at 5.3 s, released at 5.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 1: goal on(red, plate): SUCCESS after 117 planning steps (11.7 s).
  Progress: above_object at 1.1 s, at_object at 8.1 s, grasped at 8.2 s, lifted at 9.0 s, over_target at 9.9 s, lowered at 11.1 s, opened at 11.3 s, released at 11.7 s.
  Surprise: my predictions held throughout.
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=2.98, extra_mass_kg=0.303, sag_mm=0.2).
  What I changed: at 9.9 s, expect the extra weight (extra_mass_kg=0.3).
  What I changed: at 10.2 s, expect the extra weight (extra_mass_kg=0.299).
  Ground truth: no disturbance.

Episode 2: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 0.9 s, at_object at 2.3 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 3: goal on(red, plate): SUCCESS after 63 planning steps (6.3 s).
  Progress: above_object at 1.2 s, at_object at 2.0 s, grasped at 2.6 s, lifted at 3.2 s, over_target at 4.2 s, lowered at 5.6 s, opened at 5.9 s, released at 6.3 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 4: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.7 s, over_target at 4.2 s, lowered at 5.5 s, opened at 5.8 s, released at 6.2 s.
  Surprise: my predictions failed at 1 steps, most at 3.3 s (surprise 70; spikes start at 48).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 0.55 (onset=33, camera_offset_mm=[-4.7, -13.2, -1.3], seen_jump=[0.0047, 0.0132, 0.0013]).
  Ground truth: no disturbance.

Episode 5: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.1 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 3.9 s, over_target at 6.0 s, lowered at 7.3 s, opened at 7.6 s, released at 8.0 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 6: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Progress: above_object at 1.5 s, at_object at 11.8 s, grasped at 11.9 s, lifted at 13.3 s, over_target at 14.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 7: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 13.4 s, at_object at 12.2 s, grasped at 12.3 s.
  Surprise: my predictions failed at 4 steps, most at 12.5 s (surprise 866; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.52).
  What I changed: at 9.9 s, carry at 50% speed.
  Ground truth: no disturbance.

Episode 8: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.3 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.8 s, over_target at 4.3 s, lowered at 5.6 s, opened at 5.8 s, released at 6.4 s.
  Surprise: my predictions failed at 3 steps, most at 0.7 s (surprise 89; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 9: goal on(red, plate): SUCCESS after 87 planning steps (8.7 s).
  Progress: above_object at 1.0 s, at_object at 4.2 s, grasped at 4.3 s, lifted at 5.0 s, over_target at 6.4 s, lowered at 7.8 s, opened at 8.0 s, released at 8.7 s.
  Surprise: my predictions failed at 1 steps, most at 8.1 s (surprise 55; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 10: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.0 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.7 s, over_target at 4.0 s, lowered at 5.5 s, opened at 5.8 s, released at 6.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 11: goal on(red, plate): SUCCESS after 85 planning steps (8.5 s).
  Progress: above_object at 1.0 s, at_object at 3.9 s, grasped at 4.0 s, lifted at 5.0 s, over_target at 6.5 s, lowered at 7.8 s, opened at 8.1 s, released at 8.5 s.
  Surprise: my predictions failed at 1 steps, most at 4.0 s (surprise 55; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 12: goal on(red, plate): SUCCESS after 71 planning steps (7.1 s).
  Progress: above_object at 1.6 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 3.6 s, over_target at 4.6 s, lowered at 6.1 s, opened at 6.3 s, released at 7.1 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 13: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.0 s, at_object at 1.6 s, grasped at 1.7 s, lifted at 2.6 s, over_target at 4.8 s, lowered at 6.4 s, opened at 6.6 s, released at 7.3 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 14: goal on(red, plate): SUCCESS after 75 planning steps (7.5 s).
  Progress: above_object at 1.0 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 4.9 s, over_target at 5.7 s, lowered at 6.9 s, opened at 7.1 s, released at 7.5 s.
  Surprise: my predictions failed at 1 steps, most at 7.3 s (surprise 132; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 15: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 16: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.2 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 3.9 s, over_target at 5.5 s, lowered at 6.8 s, opened at 7.0 s, released at 7.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 17: goal on(red, plate): SUCCESS after 121 planning steps (12.1 s).
  Progress: above_object at 0.9 s, at_object at 5.5 s, grasped at 7.0 s, lifted at 8.4 s, over_target at 10.0 s, lowered at 11.2 s, opened at 11.4 s, released at 12.1 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 18: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.7 s, over_target at 4.0 s, lowered at 5.6 s, opened at 5.8 s, released at 6.4 s.
  Surprise: my predictions failed at 1 steps, most at 5.8 s (surprise 64; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 19: goal on(red, plate): SUCCESS after 91 planning steps (9.1 s).
  Progress: above_object at 1.6 s, at_object at 4.5 s, grasped at 4.6 s, lifted at 5.4 s, over_target at 7.0 s, lowered at 8.3 s, opened at 8.5 s, released at 9.1 s.
  Surprise: my predictions failed at 1 steps, most at 8.5 s (surprise 99; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 20: goal on(red, plate): SUCCESS after 72 planning steps (7.2 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.8 s, over_target at 4.0 s, lowered at 6.2 s, opened at 6.4 s, released at 7.2 s.
  Surprise: my predictions failed at 3 steps, most at 3.8 s (surprise 6830; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=36, end=38, force_N=[23.1, 26.9, 0.0], offset_mm=[-1.5, -3.3, -0.9]).
  Ground truth: push at 3.6 s (force=[-23.078594408991457, -26.837222005850677, 0.0]).

Episode 21: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 14.6 s, at_object at 6.4 s, grasped at 6.5 s.
  Surprise: my predictions failed at 3 steps, most at 2.7 s (surprise 128; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=25, end=27, force_N=[-45.3, -4.6, 1.8], offset_mm=[-0.9, 1.7, -0.3]).
  Ground truth: push at 2.6 s (force=[46.01616507284769, 4.795855691772432, 0.0]).

Episode 22: goal on(red, plate): SUCCESS after 148 planning steps (14.8 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 6.0 s, at_object at 7.3 s, grasped at 7.5 s, lifted at 9.6 s, over_target at 12.3 s, lowered at 14.0 s, opened at 14.2 s, released at 14.8 s.
  Surprise: my predictions failed at 6 steps, most at 6.3 s (surprise 9917; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=62, end=64, force_N=[-35.7, 21.6, 0.2], offset_mm=[1.5, -1.8, 1.0]).
  What I changed: at 5.3 s, carry at 50% speed.
  Ground truth: push at 6.3 s (force=[35.959350551011255, -21.956566777307458, 0.0]).

Episode 23: goal on(red, plate): SUCCESS after 70 planning steps (7.0 s).
  Progress: above_object at 1.2 s, at_object at 2.4 s, grasped at 2.5 s, lifted at 3.2 s, over_target at 4.1 s, lowered at 6.0 s, opened at 6.2 s, released at 7.0 s.
  Surprise: my predictions failed at 4 steps, most at 5.3 s (surprise 13361; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=53, end=55, force_N=[47.0, 13.2, 0.1], offset_mm=[-2.8, -0.3, -1.3]).
  Ground truth: push at 5.3 s (force=[-46.88333296227451, -13.182722555874374, 0.0]).

Episode 24: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.5 s, over_target at 4.4 s, lowered at 6.9 s, opened at 7.2 s, released at 7.6 s.
  Surprise: my predictions failed at 3 steps, most at 4.0 s (surprise 5010; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=39, end=41, force_N=[-12.1, 27.5, -0.0], offset_mm=[0.6, -1.2, 0.8]).
  Ground truth: push at 4.0 s (force=[12.083944329303097, -27.518493588925867, 0.0]).

Episode 25: goal on(red, plate): SUCCESS after 79 planning steps (7.9 s).
  Progress: above_object at 1.2 s, at_object at 3.0 s, grasped at 3.1 s, lifted at 3.9 s, over_target at 5.9 s, lowered at 7.2 s, opened at 7.5 s, released at 7.9 s.
  Surprise: my predictions failed at 3 steps, most at 5.8 s (surprise 11178; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=58, end=60, force_N=[-43.7, -9.4, -0.0], offset_mm=[1.6, 1.4, 0.4]).
  Ground truth: push at 5.8 s (force=[43.60389360190877, 9.340547450299931, 0.0]).

Episode 26: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.1 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.7 s, over_target at 4.5 s, lowered at 5.7 s, opened at 6.0 s, released at 6.4 s.
  Surprise: my predictions failed at 4 steps, most at 2.5 s (surprise 12486; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=24, end=26, force_N=[-21.2, -41.6, 0.3], offset_mm=[0.7, 2.5, 1.2]).
  Ground truth: push at 2.5 s (force=[21.28355781425198, 42.20018974344667, 0.0]).

Episode 27: goal on(red, plate): SUCCESS after 73 planning steps (7.3 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.8 s, over_target at 4.7 s, lowered at 6.4 s, opened at 6.6 s, released at 7.3 s.
  Surprise: my predictions failed at 4 steps, most at 4.6 s (surprise 8289; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=45, end=47, force_N=[11.7, -36.3, 0.0], offset_mm=[-1.4, 4.0, -0.0]).
  Ground truth: push at 4.6 s (force=[-11.816637376833446, 36.593135166683, 0.0]).

Episode 28: goal on(red, plate): SUCCESS after 84 planning steps (8.4 s).
  Progress: above_object at 1.6 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.5 s, over_target at 5.3 s, lowered at 7.3 s, opened at 7.5 s, released at 8.4 s.
  Surprise: my predictions failed at 9 steps, most at 2.5 s (surprise 172; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=24, end=26, force_N=[-32.4, -5.5, 3.8], offset_mm=[2.3, -0.4, 0.9]).
  Ground truth: push at 2.4 s (force=[31.972743043725643, 5.749971059610389, 0.0]).

Episode 29: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 2.3 s, lifted at 3.3 s, over_target at 5.6 s, lowered at 7.1 s, opened at 7.3 s, released at 8.0 s.
  Surprise: my predictions failed at 3 steps, most at 5.1 s (surprise 9975; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=50, end=52, force_N=[25.4, 33.5, -0.3], offset_mm=[-1.1, -0.9, -1.2]).
  Ground truth: push at 5.1 s (force=[-25.468354696480795, -33.783199626513586, 0.0]).

Episode 30: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 10.7 s, at_object at 12.6 s, grasped at 12.7 s, lifted at 14.2 s.
  Surprise: my predictions failed at 7 steps, most at 5.6 s (surprise 13990; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=56, end=58, force_N=[37.2, -33.4, 0.0], offset_mm=[-1.9, 1.6, -1.5]).
  What I changed: at 4.1 s, carry at 50% speed.
  Ground truth: push at 5.6 s (force=[-37.1879651314092, 33.338839733062635, 0.0]).

Episode 31: goal on(red, plate): SUCCESS after 104 planning steps (10.4 s).
  Progress: above_object at 1.4 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 5.8 s, over_target at 7.8 s, lowered at 9.4 s, opened at 9.6 s, released at 10.4 s.
  Surprise: my predictions failed at 3 steps, most at 6.5 s (surprise 10058; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=63, end=65, force_N=[17.0, 39.7, 0.1], offset_mm=[0.1, -2.5, -0.7]).
  Ground truth: push at 6.3 s (force=[-16.946491428273678, -39.529817452474624, 0.0]).

Episode 32: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.1 s, at_object at 2.9 s, grasped at 3.0 s, lifted at 3.9 s, over_target at 5.2 s, lowered at 7.1 s, opened at 7.3 s, released at 8.0 s.
  Surprise: my predictions failed at 6 steps, most at 5.9 s (surprise 8034; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=59, end=61, force_N=[14.3, 35.0, 0.0], offset_mm=[-0.4, -1.7, -0.3]).
  Ground truth: push at 5.9 s (force=[-14.249305886181846, -34.988097121005616, 0.0]).

Episode 33: goal on(red, plate): SUCCESS after 63 planning steps (6.3 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.6 s, over_target at 3.6 s, lowered at 5.7 s, opened at 6.0 s, released at 6.3 s.
  Surprise: my predictions failed at 3 steps, most at 2.9 s (surprise 9158; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=29, end=31, force_N=[7.2, 39.9, 0.0], offset_mm=[-0.4, -1.2, -1.4]).
  Ground truth: push at 2.9 s (force=[-7.217851567027591, -39.85883429535791, 0.0]).

Episode 34: goal on(red, plate): SUCCESS after 70 planning steps (7.0 s).
  Progress: above_object at 1.1 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.5 s, over_target at 4.2 s, lowered at 6.3 s, opened at 6.6 s, released at 7.0 s.
  Surprise: my predictions failed at 4 steps, most at 5.2 s (surprise 6801; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[14.6, -36.4, 0.1], offset_mm=[-0.9, 3.8, -0.6]).
  Ground truth: push at 5.3 s (force=[-14.676798931125873, 36.90539800747222, 0.0]).

Episode 35: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.1 s, at_object at 1.9 s, grasped at 2.5 s, lifted at 3.5 s, over_target at 5.1 s, lowered at 7.0 s, opened at 7.2 s, released at 7.7 s.
  Surprise: my predictions failed at 3 steps, most at 5.9 s (surprise 7731; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=59, end=61, force_N=[-34.1, 15.0, -0.0], offset_mm=[1.0, -0.4, 1.6]).
  Ground truth: push at 5.9 s (force=[34.01070499148584, -14.961054212462347, 0.0]).

Episode 36: goal on(red, plate): SUCCESS after 63 planning steps (6.3 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.7 s, over_target at 4.5 s, lowered at 5.6 s, opened at 5.9 s, released at 6.3 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 7373; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[32.8, 15.8, -0.0], offset_mm=[-1.6, -0.5, -0.7]).
  Ground truth: push at 5.3 s (force=[-32.81888686057621, -15.830473665954154, 0.0]).

Episode 37: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 8.5 s, at_object at 13.2 s, grasped at 13.3 s, lifted at 14.3 s.
  Surprise: my predictions failed at 7 steps, most at 4.7 s (surprise 7839; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=47, end=49, force_N=[19.9, -32.1, 0.0], offset_mm=[-0.4, 0.8, -0.2]).
  What I changed: at 3.8 s, carry at 50% speed.
  Ground truth: push at 4.8 s (force=[-19.850645061690944, 32.20622405527275, 0.0]).

Episode 38: goal on(red, plate): SUCCESS after 89 planning steps (8.9 s).
  Progress: above_object at 1.8 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 4.9 s, over_target at 6.2 s, lowered at 8.2 s, opened at 8.5 s, released at 8.9 s.
  Surprise: my predictions failed at 3 steps, most at 5.2 s (surprise 6277; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=54, force_N=[-26.7, 22.0, 0.1], offset_mm=[1.7, -1.5, -1.1]).
  Ground truth: push at 5.2 s (force=[26.65387221960392, -21.972720288144398, 0.0]).

Episode 39: goal on(red, plate): SUCCESS after 58 planning steps (5.8 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.6 s, over_target at 3.6 s, lowered at 4.8 s, opened at 5.0 s, released at 5.8 s.
  Surprise: my predictions failed at 3 steps, most at 5.0 s (surprise 11967; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=48, end=50, force_N=[-40.1, -23.4, 0.3], offset_mm=[1.4, 1.1, 1.7]).
  Ground truth: push at 4.9 s (force=[40.3022527572971, 23.49935978956869, 0.0]).

Episode 40: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.2 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.8 s, over_target at 3.9 s, lowered at 5.4 s, opened at 5.7 s, released at 6.2 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 310; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=38, end=51, force_N=[-0.1, -0.1, 5.3], offset_mm=[-0.4, 0.2, 0.1]).
  What I changed: at 2.5 s, expect the extra weight (extra_mass_kg=0.531).
  What I changed: at 2.8 s, expect the extra weight (extra_mass_kg=0.531).
  Ground truth: payload at 0.0 s (mass=0.536129492246605).

Episode 41: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 14.5 s, at_object at 9.6 s, grasped at 11.9 s, lifted at 13.6 s.
  Surprise: my predictions failed at 5 steps, most at 14.8 s (surprise 36641; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=4.46).
  What I changed: at 13.7 s, carry at 50% speed.
  Ground truth: payload at 0.0 s (mass=0.37181083289788563).

Episode 42: goal on(red, plate): SUCCESS after 115 planning steps (11.5 s).
  Progress: above_object at 1.2 s, at_object at 5.4 s, grasped at 5.5 s, lifted at 6.8 s, over_target at 9.6 s, lowered at 10.8 s, opened at 11.0 s, released at 11.5 s.
  Surprise: my predictions failed at 4 steps, most at 6.2 s (surprise 90; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.35, extra_mass_kg=0.545, sag_mm=0.77).
  What I changed: at 6.2 s, expect the extra weight (extra_mass_kg=0.554).
  What I changed: at 6.5 s, expect the extra weight (extra_mass_kg=0.555).
  Ground truth: payload at 0.0 s (mass=0.5629452692432111).

Episode 43: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.1 s, at_object at 1.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: payload at 0.0 s (mass=0.3175704104415583).

Episode 44: goal on(red, plate): SUCCESS after 123 planning steps (12.3 s).
  Progress: above_object at 1.8 s, at_object at 7.7 s, grasped at 7.8 s, lifted at 8.6 s, over_target at 10.1 s, lowered at 11.4 s, opened at 11.6 s, released at 12.3 s.
  Surprise: my predictions failed at 2 steps, most at 8.0 s (surprise 81; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.97, extra_mass_kg=0.405, sag_mm=0.08).
  What I changed: at 8.0 s, expect the extra weight (extra_mass_kg=0.409).
  What I changed: at 8.3 s, expect the extra weight (extra_mass_kg=0.402).
  Ground truth: payload at 0.0 s (mass=0.4008351181636981).

Episode 45: goal on(red, plate): SUCCESS after 71 planning steps (7.1 s).
  Progress: above_object at 1.4 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.5 s, over_target at 5.2 s, lowered at 6.4 s, opened at 6.6 s, released at 7.1 s.
  Surprise: my predictions failed at 2 steps, most at 3.0 s (surprise 65; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.43, extra_mass_kg=0.35, sag_mm=0.27).
  What I changed: at 3.0 s, expect the extra weight (extra_mass_kg=0.348).
  What I changed: at 3.3 s, expect the extra weight (extra_mass_kg=0.343).
  Ground truth: payload at 0.0 s (mass=0.3450838400684517).

Episode 46: goal on(red, plate): SUCCESS after 96 planning steps (9.6 s).
  Progress: above_object at 1.8 s, at_object at 4.2 s, grasped at 4.3 s, lifted at 5.9 s, over_target at 7.6 s, lowered at 9.0 s, opened at 9.2 s, released at 9.6 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 101; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.33, extra_mass_kg=0.441, sag_mm=0.02).
  What I changed: at 5.3 s, expect the extra weight (extra_mass_kg=0.44).
  What I changed: at 5.6 s, expect the extra weight (extra_mass_kg=0.429).
  Ground truth: payload at 0.0 s (mass=0.4351018099947861).

Episode 47: goal on(red, plate): SUCCESS after 72 planning steps (7.2 s).
  Progress: above_object at 1.1 s, at_object at 2.0 s, grasped at 2.1 s, lifted at 2.9 s, over_target at 4.1 s, lowered at 6.3 s, opened at 6.5 s, released at 7.2 s.
  Surprise: my predictions failed at 13 steps, most at 5.2 s (surprise 382; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=52, end=62, force_N=[-0.8, 0.0, -1.8], offset_mm=[-0.1, -0.0, 0.3]).
  What I changed: at 2.2 s, expect the extra weight (extra_mass_kg=0.504).
  What I changed: at 2.5 s, expect the extra weight (extra_mass_kg=0.53).
  Ground truth: payload at 0.0 s (mass=0.5388972810861883).

Episode 48: goal on(red, plate): SUCCESS after 99 planning steps (9.9 s).
  Progress: above_object at 1.5 s, at_object at 4.6 s, grasped at 4.7 s, lifted at 6.0 s, over_target at 7.4 s, lowered at 8.9 s, opened at 9.1 s, released at 9.9 s.
  Surprise: my predictions failed at 32 steps, most at 5.9 s (surprise 248; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=1.29).
  What I changed: at 8.2 s, expect the extra weight (extra_mass_kg=0.372).
  What I changed: at 8.5 s, expect the extra weight (extra_mass_kg=0.375).
  Ground truth: payload at 0.0 s (mass=0.36919266269812423).

Episode 49: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 11.9 s, at_object at 12.7 s, grasped at 12.9 s, lifted at 14.3 s, over_target at 14.8 s.
  Surprise: my predictions failed at 10 steps, most at 13.1 s (surprise 927; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.98).
  What I changed: at 2.2 s, carry at 50% speed.
  What I changed: at 6.3 s, expect the extra weight (extra_mass_kg=0.317).
  What I changed: at 6.6 s, expect the extra weight (extra_mass_kg=0.315).
  Ground truth: payload at 0.0 s (mass=0.31560639031932286).

Episode 50: goal on(red, plate): SUCCESS after 62 planning steps (6.2 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.5 s, over_target at 3.8 s, lowered at 5.2 s, opened at 5.4 s, released at 6.2 s.
  Surprise: my predictions failed at 2 steps, most at 5.5 s (surprise 135; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.21, extra_mass_kg=0.429, sag_mm=0.0).
  What I changed: at 2.0 s, expect the extra weight (extra_mass_kg=0.423).
  What I changed: at 2.3 s, expect the extra weight (extra_mass_kg=0.411).
  Ground truth: payload at 0.0 s (mass=0.4213655519464584).

Episode 51: goal on(red, plate): SUCCESS after 107 planning steps (10.7 s).
  Progress: above_object at 1.0 s, at_object at 5.8 s, grasped at 5.9 s, lifted at 6.6 s, over_target at 8.0 s, lowered at 9.9 s, opened at 10.1 s, released at 10.7 s.
  Surprise: my predictions failed at 2 steps, most at 6.1 s (surprise 70; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.45, extra_mass_kg=0.352, sag_mm=0.36).
  What I changed: at 6.2 s, expect the extra weight (extra_mass_kg=0.352).
  What I changed: at 6.5 s, expect the extra weight (extra_mass_kg=0.347).
  Ground truth: payload at 0.0 s (mass=0.3595539133527766).

Episode 52: goal on(red, plate): SUCCESS after 106 planning steps (10.6 s).
  Progress: above_object at 1.0 s, at_object at 6.6 s, grasped at 6.7 s, lifted at 7.5 s, over_target at 8.3 s, lowered at 9.9 s, opened at 10.2 s, released at 10.6 s.
  Surprise: my predictions failed at 3 steps, most at 7.1 s (surprise 54; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.2, extra_mass_kg=0.326, sag_mm=0.48).
  What I changed: at 7.1 s, expect the extra weight (extra_mass_kg=0.326).
  What I changed: at 7.4 s, expect the extra weight (extra_mass_kg=0.32).
  Ground truth: payload at 0.0 s (mass=0.32722591368573656).

Episode 53: goal on(red, plate): SUCCESS after 127 planning steps (12.7 s).
  Progress: above_object at 1.1 s, at_object at 7.9 s, grasped at 8.0 s, lifted at 8.8 s, over_target at 10.0 s, lowered at 11.7 s, opened at 11.9 s, released at 12.7 s.
  Surprise: my predictions failed at 1 steps, most at 8.4 s (surprise 117; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.69, extra_mass_kg=0.478, sag_mm=0.12).
  What I changed: at 8.4 s, expect the extra weight (extra_mass_kg=0.477).
  What I changed: at 8.7 s, expect the extra weight (extra_mass_kg=0.469).
  Ground truth: payload at 0.0 s (mass=0.47409971579605514).

Episode 54: goal on(red, plate): SUCCESS after 58 planning steps (5.8 s).
  Progress: above_object at 1.1 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 3.0 s, over_target at 3.8 s, lowered at 5.1 s, opened at 5.3 s, released at 5.8 s.
  Surprise: my predictions failed at 2 steps, most at 2.6 s (surprise 76; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.83, extra_mass_kg=0.39, sag_mm=0.03).
  What I changed: at 2.6 s, expect the extra weight (extra_mass_kg=0.385).
  What I changed: at 2.9 s, expect the extra weight (extra_mass_kg=0.38).
  Ground truth: payload at 0.0 s (mass=0.38960883984567674).

Episode 55: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s.
  Surprise: my predictions failed at 1 steps, most at 8.6 s (surprise 53; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: payload at 0.0 s (mass=0.5015984633869078).

Episode 56: goal on(red, plate): SUCCESS after 127 planning steps (12.7 s).
  Progress: above_object at 0.9 s, at_object at 7.8 s, grasped at 7.9 s, lifted at 8.9 s, over_target at 10.8 s, lowered at 12.0 s, opened at 12.2 s, released at 12.7 s.
  Surprise: my predictions failed at 1 steps, most at 8.3 s (surprise 71; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.57, extra_mass_kg=0.364, sag_mm=0.04).
  What I changed: at 8.3 s, expect the extra weight (extra_mass_kg=0.367).
  What I changed: at 8.6 s, expect the extra weight (extra_mass_kg=0.36).
  Ground truth: payload at 0.0 s (mass=0.35985463319046396).

Episode 57: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.8 s, at_object at 14.5 s, grasped at 14.7 s, lifted at 10.3 s, over_target at 11.0 s.
  Surprise: my predictions failed at 23 steps, most at 6.7 s (surprise 2263; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=1.59).
  What I changed: at 2.7 s, expect the extra weight (extra_mass_kg=0.57).
  What I changed: at 3.0 s, expect the extra weight (extra_mass_kg=0.565).
  What I changed: at 8.4 s, carry at 50% speed.
  Ground truth: payload at 0.0 s (mass=0.5826339331519493).

Episode 58: goal on(red, plate): SUCCESS after 108 planning steps (10.8 s).
  Progress: above_object at 1.2 s, at_object at 6.2 s, grasped at 6.3 s, lifted at 7.1 s, over_target at 8.3 s, lowered at 9.8 s, opened at 10.0 s, released at 10.8 s.
  Surprise: my predictions failed at 2 steps, most at 10.1 s (surprise 125; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.11, extra_mass_kg=0.419, sag_mm=0.0).
  What I changed: at 6.9 s, expect the extra weight (extra_mass_kg=0.404).
  What I changed: at 7.2 s, expect the extra weight (extra_mass_kg=0.398).
  Ground truth: payload at 0.0 s (mass=0.40953305047344857).

Episode 59: goal on(red, plate): SUCCESS after 79 planning steps (7.9 s).
  Progress: above_object at 1.7 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 4.3 s, over_target at 5.5 s, lowered at 7.1 s, opened at 7.3 s, released at 7.9 s.
  Surprise: my predictions failed at 3 steps, most at 7.3 s (surprise 71; spikes start at 48).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.31, extra_mass_kg=0.337, sag_mm=0.0).
  What I changed: at 3.8 s, expect the extra weight (extra_mass_kg=0.33).
  What I changed: at 4.1 s, expect the extra weight (extra_mass_kg=0.324).
  Ground truth: payload at 0.0 s (mass=0.3316485838710688).

Episode 60: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 11.4 s, at_object at 13.4 s, grasped at 13.6 s, lifted at 14.8 s, over_target at 9.2 s.
  Surprise: my predictions failed at 3 steps, most at 10.6 s (surprise 196; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=106, sink_mm=60.2, slide_mm=[-8.0, -0.3], weight_lost_N=-0.0).
  What I changed: at 10.6 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 61: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.5 s, at_object at 11.4 s, grasped at 11.5 s, lifted at 6.3 s.
  Surprise: my predictions failed at 4 steps, most at 8.2 s (surprise 28518; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=82, end=83, force_N=[4.4, 1.1, -88.2], offset_mm=[2.6, 1.3, 3.3]).
  What I changed: at 6.7 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 62: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 5.2 s, at_object at 14.8 s, grasped at 14.9 s, lifted at 2.6 s.
  Surprise: my predictions failed at 5 steps, most at 5.7 s (surprise 217; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=57, end=60, force_N=[68.4, -3.4, -165.1], offset_mm=[2.0, 0.4, 2.3]).
  What I changed: at 3.0 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 63: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 14.2 s, at_object at 12.0 s, grasped at 12.1 s, lifted at 9.4 s.
  Surprise: my predictions failed at 4 steps, most at 4.3 s (surprise 4264; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=43, sink_mm=163.1, slide_mm=[-22.5, 53.3], weight_lost_N=0.64).
  What I changed: at 4.3 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 64: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 5.0 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.6 s.
  Surprise: my predictions failed at 9 steps, most at 3.6 s (surprise 2474; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=36, sink_mm=151.2, slide_mm=[-41.4, 33.2], weight_lost_N=0.63).
  What I changed: at 3.6 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 65: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.3 s.
  Surprise: my predictions failed at 2 steps, most at 5.8 s (surprise 71; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 66: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 7.9 s, grasped at 8.0 s.
  Surprise: my predictions failed at 14 steps, most at 8.4 s (surprise 1140; spikes start at 48).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=0.78).
  What I changed: at 8.1 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 67: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 11.7 s, at_object at 13.5 s, grasped at 13.6 s, lifted at 14.8 s, over_target at 9.2 s.
  Surprise: my predictions failed at 6 steps, most at 9.6 s (surprise 1698; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=96, sink_mm=100.7, slide_mm=[-11.6, 23.5], weight_lost_N=0.63).
  What I changed: at 9.5 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 68: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.2 s, at_object at 13.5 s, grasped at 13.6 s, lifted at 14.7 s, over_target at 9.9 s.
  Surprise: my predictions failed at 7 steps, most at 6.4 s (surprise 4120; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=64, sink_mm=164.5, slide_mm=[-14.0, -7.7], weight_lost_N=0.61).
  What I changed: at 6.6 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 69: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 13.7 s, at_object at 11.7 s, grasped at 11.8 s.
  Surprise: my predictions failed at 5 steps, most at 12.0 s (surprise 569; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=120, sink_mm=30.3, slide_mm=[-258.2, -177.3], weight_lost_N=0.49).
  What I changed: at 12.0 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 70: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 10.4 s, at_object at 14.6 s, grasped at 14.7 s, lifted at 3.0 s.
  Surprise: my predictions failed at 8 steps, most at 4.1 s (surprise 3570; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=41, sink_mm=155.6, slide_mm=[35.3, 35.3], weight_lost_N=0.72).
  What I changed: at 4.1 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 71: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 8.9 s, at_object at 7.3 s, grasped at 7.4 s.
  Surprise: my predictions failed at 3 steps, most at 5.2 s (surprise 380; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=52, sink_mm=6.2, slide_mm=[-349.5, -35.7], weight_lost_N=-0.0).
  What I changed: at 5.2 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 72: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 14.0 s, grasped at 14.3 s, lifted at 7.9 s, over_target at 8.4 s.
  Surprise: my predictions failed at 10 steps, most at 13.1 s (surprise 94219; spikes start at 48).
  My explanation: something pushed my arm, probability 1.00 (onset=131, end=131, force_N=[8.3, 1.8, -131.6], offset_mm=[1.6, 0.8, 1.8]).
  What I changed: at 4.8 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 73: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.3 s, at_object at 11.3 s, grasped at 11.4 s, lifted at 3.6 s, over_target at 5.3 s.
  Surprise: my predictions failed at 5 steps, most at 5.3 s (surprise 3965; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=53, sink_mm=142.0, slide_mm=[0.1, 12.6], weight_lost_N=0.66).
  What I changed: at 5.3 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 74: goal on(red, plate): SUCCESS after 124 planning steps (12.4 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 7.7 s, at_object at 8.4 s, grasped at 8.6 s, lifted at 9.7 s, over_target at 10.2 s, lowered at 11.6 s, opened at 11.8 s, released at 12.4 s.
  Surprise: my predictions failed at 3 steps, most at 4.4 s (surprise 332; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=44, sink_mm=119.7, slide_mm=[-20.7, 4.3], weight_lost_N=0.67).
  What I changed: at 4.5 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 75: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 6.3 s, at_object at 2.4 s, grasped at 2.7 s, lifted at 3.7 s.
  Surprise: my predictions failed at 1 steps, most at 5.4 s (surprise 6049; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=54, sink_mm=157.3, slide_mm=[-27.1, -69.2], weight_lost_N=0.67).
  What I changed: at 5.4 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 76: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 10.7 s, at_object at 12.1 s, grasped at 12.4 s, lifted at 13.6 s, over_target at 14.5 s.
  Surprise: my predictions failed at 7 steps, most at 9.7 s (surprise 4829; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=97, sink_mm=154.9, slide_mm=[-23.6, -2.8], weight_lost_N=0.64).
  What I changed: at 9.7 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 77: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 8.5 s, at_object at 12.9 s, grasped at 13.0 s, lifted at 6.9 s.
  Surprise: my predictions failed at 3 steps, most at 13.1 s (surprise 245; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=131, sink_mm=6.8, slide_mm=[-120.5, -162.3], weight_lost_N=-0.0).
  What I changed: at 13.1 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 78: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 11.6 s, at_object at 13.4 s, grasped at 13.6 s, lifted at 14.9 s, over_target at 11.1 s.
  Surprise: my predictions failed at 7 steps, most at 11.2 s (surprise 4910; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=112, sink_mm=162.8, slide_mm=[-8.9, 18.5], weight_lost_N=0.66).
  What I changed: at 11.2 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 79: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 11.9 s, at_object at 13.4 s, grasped at 13.6 s, lifted at 2.7 s, over_target at 3.7 s.
  Surprise: my predictions failed at 2 steps, most at 4.0 s (surprise 2557; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=40, sink_mm=127.9, slide_mm=[-7.1, -4.9], weight_lost_N=0.58).
  What I changed: at 4.0 s, carry at 50% speed.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 80: goal on(red, plate): SUCCESS after 70 planning steps (7.0 s).
  Progress: above_object at 1.1 s, at_object at 3.1 s, grasped at 3.2 s, lifted at 3.9 s, over_target at 5.0 s, lowered at 6.2 s, opened at 6.4 s, released at 7.0 s.
  Surprise: my predictions failed at 3 steps, most at 4.3 s (surprise 105; spikes start at 48).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=43, camera_offset_mm=[2.4, 16.8, -0.9], seen_jump=[-0.0024, -0.0168, 0.0009]).
  Ground truth: camera_shift at 2.5 s (offset=[-0.039812385659612884, -0.04192262101013534, 0.009091809873814744]).

Episode 81: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 9.4 s, at_object at 10.6 s, grasped at 10.7 s, lifted at 11.9 s, over_target at 13.6 s.
  Surprise: my predictions failed at 3 steps, most at 7.7 s (surprise 1638; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=77, sink_mm=156.3, slide_mm=[11.7, -135.8], weight_lost_N=0.6).
  What I changed: at 7.7 s, carry at 50% speed.
  Ground truth: camera_shift at 2.6 s (offset=[-0.04275684958417764, 2.798957440961015e-05, 0.0024042690403075564]).

Episode 82: goal on(red, plate): SUCCESS after 106 planning steps (10.6 s).
  Progress: above_object at 3.1 s, at_object at 3.8 s, grasped at 4.6 s, lifted at 5.3 s, over_target at 8.8 s, lowered at 9.9 s, opened at 10.2 s, released at 10.6 s.
  Surprise: my predictions failed at 1 steps, most at 5.1 s (surprise 74; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 0.9 s (offset=[0.058440562524408976, -0.001801098304528124, 0.005154576906165829]).

Episode 83: goal on(red, plate): SUCCESS after 69 planning steps (6.9 s).
  Progress: above_object at 1.0 s, at_object at 2.7 s, grasped at 2.8 s, lifted at 3.6 s, over_target at 4.6 s, lowered at 6.2 s, opened at 6.5 s, released at 6.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.7 s (offset=[-0.04587334934213697, 0.0007429233657669993, 0.005715714014276148]).

Episode 84: goal on(red, plate): SUCCESS after 81 planning steps (8.1 s).
  Progress: above_object at 1.0 s, at_object at 3.5 s, grasped at 3.6 s, lifted at 4.4 s, over_target at 5.8 s, lowered at 7.2 s, opened at 7.4 s, released at 8.1 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 3.9 s (offset=[-0.044730906450852495, 0.02658450569052531, 0.008641193732267566]).

Episode 85: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.1 s, at_object at 1.8 s, grasped at 1.9 s, lifted at 2.7 s, over_target at 4.9 s, lowered at 6.8 s, opened at 7.0 s, released at 7.7 s.
  Surprise: my predictions failed at 1 steps, most at 7.1 s (surprise 127; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 3.8 s (offset=[0.03892311737917909, 0.03428607415638614, 0.008548478572491197]).

Episode 86: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 4 time(s) and went back to re-grasp it.
  Progress: above_object at 14.9 s, at_object at 13.0 s, grasped at 13.1 s.
  Surprise: my predictions failed at 6 steps, most at 13.4 s (surprise 794; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=134, sink_mm=16.9, slide_mm=[-131.9, -309.1], weight_lost_N=0.46).
  What I changed: at 3.2 s, carry at 50% speed.
  Ground truth: camera_shift at 1.3 s (offset=[0.029825130379335377, -0.0060932417200100985, 0.009623900801326886]).

Episode 87: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.0 s, at_object at 2.1 s, grasped at 2.5 s, lifted at 4.1 s, over_target at 5.4 s, lowered at 6.8 s, opened at 7.0 s, released at 7.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 4.4 s (offset=[0.03322485056448212, -0.009154353346265796, 0.009452576276459101]).

Episode 88: goal on(red, plate): SUCCESS after 110 planning steps (11.0 s).
  Progress: above_object at 1.3 s, at_object at 5.7 s, grasped at 5.8 s, lifted at 6.6 s, over_target at 8.5 s, lowered at 10.0 s, opened at 10.2 s, released at 11.0 s.
  Surprise: my predictions failed at 5 steps, most at 6.8 s (surprise 122; spikes start at 48).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=58, camera_offset_mm=[32.7, 31.8, -0.5], seen_jump=[-0.0327, -0.0318, 0.0005]).
  Ground truth: camera_shift at 1.8 s (offset=[0.04211078077803654, -0.034865798038288626, -0.005352541607213923]).

Episode 89: goal on(red, plate): SUCCESS after 65 planning steps (6.5 s).
  Progress: above_object at 1.0 s, at_object at 1.8 s, grasped at 2.0 s, lifted at 2.8 s, over_target at 4.2 s, lowered at 5.7 s, opened at 5.9 s, released at 6.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.8 s (offset=[0.01847932739603208, -0.05466704589629126, -0.004677394554154148]).

Episode 90: goal on(red, plate): SUCCESS after 64 planning steps (6.4 s).
  Progress: above_object at 1.0 s, at_object at 1.9 s, grasped at 2.0 s, lifted at 2.8 s, over_target at 4.6 s, lowered at 5.7 s, opened at 6.0 s, released at 6.4 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.4 s (offset=[-0.041993910668189416, -0.010483015775755547, -0.009189785776231307]).

Episode 91: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 11.4 s, at_object at 4.4 s, grasped at 5.8 s, lifted at 6.8 s, over_target at 9.0 s.
  Surprise: my predictions failed at 4 steps, most at 9.3 s (surprise 3173; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=93, sink_mm=152.0, slide_mm=[134.5, -37.6], weight_lost_N=0.61).
  What I changed: at 6.3 s, expect the extra weight (extra_mass_kg=0.298).
  What I changed: at 6.6 s, expect the extra weight (extra_mass_kg=0.297).
  Ground truth: camera_shift at 4.7 s (offset=[-0.0054638965435530274, -0.048121998264614184, -0.009432692697729578]).

Episode 92: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.1 s, at_object at 3.0 s, grasped at 3.1 s, lifted at 4.0 s, over_target at 5.6 s, lowered at 7.0 s, opened at 7.3 s, released at 7.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.2 s (offset=[-0.0058580413491001, -0.029911513277774877, 0.00025517446524156093]).

Episode 93: goal on(red, plate): SUCCESS after 60 planning steps (6.0 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.7 s, over_target at 3.9 s, lowered at 5.1 s, opened at 5.3 s, released at 6.0 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 4.0 s (offset=[0.028861506665298697, -0.01378013548003836, 0.006826345592247663]).

Episode 94: goal on(red, plate): SUCCESS after 60 planning steps (6.0 s).
  Progress: above_object at 1.0 s, at_object at 1.7 s, grasped at 1.8 s, lifted at 2.7 s, over_target at 3.6 s, lowered at 5.0 s, opened at 5.2 s, released at 6.0 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.7 s (offset=[0.03684024207957708, 0.016408807108357116, 0.009321241615681404]).

Episode 95: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 3.4 s, at_object at 8.7 s, grasped at 8.8 s, lifted at 9.8 s, over_target at 12.3 s, lowered at 14.9 s.
  Surprise: my predictions failed at 5 steps, most at 2.2 s (surprise 275; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=22, sink_mm=8.0, slide_mm=[-121.0, -174.9], weight_lost_N=-0.0).
  What I changed: at 2.2 s, carry at 50% speed.
  Ground truth: camera_shift at 2.6 s (offset=[-0.03491547804767095, -0.014393590596098065, -0.005166485718113101]).

Episode 96: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.0 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.5 s, over_target at 5.1 s, lowered at 6.8 s, opened at 7.0 s, released at 7.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 1.3 s (offset=[0.028057332016747115, -0.02377533154505764, -0.004233384859848448]).

Episode 97: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 3 time(s) and went back to re-grasp it.
  Progress: above_object at 12.7 s, at_object at 13.6 s, grasped at 14.2 s.
  Surprise: my predictions failed at 5 steps, most at 9.6 s (surprise 342; spikes start at 48).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=96, sink_mm=15.0, slide_mm=[-225.9, -139.9], weight_lost_N=-0.0).
  What I changed: at 6.7 s, carry at 50% speed.
  Ground truth: camera_shift at 1.3 s (offset=[-0.03996166319853981, -0.024015474895515043, 0.006194215518255554]).

Episode 98: goal on(red, plate): SUCCESS after 104 planning steps (10.4 s).
  Progress: above_object at 1.0 s, at_object at 6.3 s, grasped at 6.4 s, lifted at 7.0 s, over_target at 7.9 s, lowered at 9.4 s, opened at 9.6 s, released at 10.4 s.
  Surprise: my predictions failed at 1 steps, most at 9.7 s (surprise 122; spikes start at 48).
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.7 s (offset=[-0.03589560013521975, -0.014336394042755926, 0.006362419419418208]).

Episode 99: goal on(red, plate): SUCCESS after 65 planning steps (6.5 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 2.9 s, over_target at 3.9 s, lowered at 5.8 s, opened at 6.1 s, released at 6.5 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: camera_shift at 2.5 s (offset=[-0.04116309146536361, -0.04194981037048178, -0.0026119117781663826]).
```
