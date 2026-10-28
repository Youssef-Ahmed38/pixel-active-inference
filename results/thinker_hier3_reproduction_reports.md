# Episode reports (the agent's own account vs ground truth)

```
Episode 0: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 2.9 s, over_target at 4.8 s, lowered at 7.4 s, opened at 7.6 s, released at 8.0 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 1: goal on(red, plate): SUCCESS after 79 planning steps (7.9 s).
  Progress: above_object at 1.2 s, at_object at 3.6 s, grasped at 3.9 s, lifted at 4.8 s, over_target at 5.7 s, lowered at 7.1 s, opened at 7.3 s, released at 7.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 2: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s, at_object at 11.3 s, grasped at 11.4 s, lifted at 12.1 s, over_target at 13.3 s, lowered at 14.6 s, opened at 14.8 s.
  Surprise: my predictions held throughout.
  My explanation: something pushed my arm, probability 1.00 (onset=95, end=97, force_N=[22.5, -9.0, -33.2], offset_mm=[-0.4, -0.2, -0.1]).
  Ground truth: no disturbance.

Episode 3: goal on(red, plate): SUCCESS after 92 planning steps (9.2 s).
  Progress: above_object at 1.1 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 4.4 s, over_target at 5.3 s, lowered at 8.5 s, opened at 8.7 s, released at 9.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 4: goal on(red, plate): SUCCESS after 89 planning steps (8.9 s).
  Progress: above_object at 1.2 s, at_object at 4.0 s, grasped at 4.1 s, lifted at 4.8 s, over_target at 6.5 s, lowered at 8.2 s, opened at 8.4 s, released at 8.9 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 5: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.6 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.5 s, over_target at 5.4 s, lowered at 7.0 s, opened at 7.2 s, released at 7.7 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 6: goal on(red, plate): SUCCESS after 141 planning steps (14.1 s).
  Progress: above_object at 1.2 s, at_object at 10.0 s, grasped at 10.1 s, lifted at 10.7 s, over_target at 12.1 s, lowered at 13.5 s, opened at 13.7 s, released at 14.1 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened, probability 1.00.
  Ground truth: no disturbance.

Episode 7: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.3 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.4 s, over_target at 5.6 s, lowered at 6.9 s, opened at 7.1 s, released at 7.6 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: no disturbance.

Episode 8: goal on(red, plate): SUCCESS after 59 planning steps (5.9 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 2.9 s, over_target at 3.9 s, lowered at 5.2 s, opened at 5.4 s, released at 5.9 s.
  Surprise: my predictions failed at 3 steps, most at 3.6 s (surprise 7042; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=36, end=38, force_N=[23.1, 26.9, 0.1], offset_mm=[-0.7, -0.7, -0.8]).
  Ground truth: push at 3.6 s (force=[-23.078594408991457, -26.837222005850677, 0.0]).

Episode 9: goal on(red, plate): SUCCESS after 94 planning steps (9.4 s).
  Progress: above_object at 1.2 s, at_object at 5.1 s, grasped at 5.2 s, lifted at 5.9 s, over_target at 7.0 s, lowered at 8.7 s, opened at 8.9 s, released at 9.4 s.
  Surprise: my predictions failed at 2 steps, most at 2.7 s (surprise 650; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=25, end=27, force_N=[-46.1, -4.8, 0.7], offset_mm=[1.4, 0.6, 0.7]).
  Ground truth: push at 2.6 s (force=[46.01616507284769, 4.795855691772432, 0.0]).

Episode 10: goal on(red, plate): SUCCESS after 115 planning steps (11.5 s).
  Progress: above_object at 1.0 s, at_object at 7.6 s, grasped at 7.7 s, lifted at 8.3 s, over_target at 9.4 s, lowered at 10.8 s, opened at 11.0 s, released at 11.5 s.
  Surprise: my predictions failed at 1 steps, most at 6.2 s (surprise 128; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=62, end=64, force_N=[-35.7, 21.4, 5.0], offset_mm=[1.4, -1.0, 0.7]).
  Ground truth: push at 6.3 s (force=[35.959350551011255, -21.956566777307458, 0.0]).

Episode 11: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.1 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 4.4 s, over_target at 5.3 s, lowered at 6.8 s, opened at 7.0 s, released at 7.6 s.
  Surprise: my predictions failed at 3 steps, most at 5.3 s (surprise 13289; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=53, end=55, force_N=[46.9, 13.2, 0.0], offset_mm=[-1.9, -1.4, -1.8]).
  Ground truth: push at 5.3 s (force=[-46.88333296227451, -13.182722555874374, 0.0]).

Episode 12: goal on(red, plate): SUCCESS after 90 planning steps (9.0 s).
  Progress: above_object at 1.2 s, at_object at 4.3 s, grasped at 4.6 s, lifted at 5.3 s, over_target at 6.9 s, lowered at 8.4 s, opened at 8.6 s, released at 9.0 s.
  Surprise: my predictions failed at 3 steps, most at 3.9 s (surprise 382; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=39, end=41, force_N=[-12.0, 27.4, 0.7], offset_mm=[0.3, -0.7, -0.2]).
  Ground truth: push at 4.0 s (force=[12.083944329303097, -27.518493588925867, 0.0]).

Episode 13: goal on(red, plate): SUCCESS after 74 planning steps (7.4 s).
  Progress: above_object at 1.6 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.5 s, over_target at 5.4 s, lowered at 6.7 s, opened at 6.9 s, released at 7.4 s.
  Surprise: my predictions failed at 3 steps, most at 5.8 s (surprise 11137; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=58, end=60, force_N=[-43.6, -9.4, -0.0], offset_mm=[1.6, 0.5, 1.4]).
  Ground truth: push at 5.8 s (force=[43.60389360190877, 9.340547450299931, 0.0]).

Episode 14: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.2 s, at_object at 3.5 s, grasped at 3.6 s, lifted at 4.3 s, over_target at 5.5 s, lowered at 7.3 s, opened at 7.5 s, released at 8.0 s.
  Surprise: my predictions failed at 3 steps, most at 2.5 s (surprise 117; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=24, end=26, force_N=[-20.9, -41.7, 3.3], offset_mm=[0.9, 1.9, 0.1]).
  Ground truth: push at 2.5 s (force=[21.28355781425198, 42.20018974344667, 0.0]).

Episode 15: goal on(red, plate): SUCCESS after 88 planning steps (8.8 s).
  Progress: above_object at 1.3 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.4 s, over_target at 6.7 s, lowered at 8.1 s, opened at 8.3 s, released at 8.8 s.
  Surprise: my predictions failed at 3 steps, most at 4.6 s (surprise 8241; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=45, end=47, force_N=[11.8, -36.4, -0.0], offset_mm=[-0.0, 1.6, -0.3]).
  Ground truth: push at 4.6 s (force=[-11.816637376833446, 36.593135166683, 0.0]).

Episode 16: goal on(red, plate): SUCCESS after 61 planning steps (6.1 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 3.0 s, over_target at 4.0 s, lowered at 5.5 s, opened at 5.7 s, released at 6.1 s.
  Surprise: my predictions failed at 1 steps, most at 3.0 s (surprise 47; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.02, extra_mass_kg=0.308, sag_mm=0.08).
  What I changed: at 3.0 s, expect the extra weight (extra_mass_kg=0.299).
  What I changed: at 3.3 s, expect the extra weight (extra_mass_kg=0.296).
  Ground truth: payload at 0.0 s (mass=0.30849590134363886).

Episode 17: goal on(red, plate): SUCCESS after 86 planning steps (8.6 s).
  Progress: above_object at 1.2 s, at_object at 4.8 s, grasped at 5.1 s, lifted at 5.8 s, over_target at 6.7 s, lowered at 8.0 s, opened at 8.2 s, released at 8.6 s.
  Surprise: my predictions failed at 1 steps, most at 5.8 s (surprise 60; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=3.3, extra_mass_kg=0.336, sag_mm=0.02).
  What I changed: at 5.8 s, expect the extra weight (extra_mass_kg=0.336).
  What I changed: at 6.1 s, expect the extra weight (extra_mass_kg=0.333).
  Ground truth: payload at 0.0 s (mass=0.3372849829498692).

Episode 18: goal on(red, plate): SUCCESS after 103 planning steps (10.3 s).
  Progress: above_object at 1.0 s, at_object at 6.6 s, grasped at 6.7 s, lifted at 7.4 s, over_target at 8.4 s, lowered at 9.7 s, opened at 9.9 s, released at 10.3 s.
  Surprise: my predictions failed at 1 steps, most at 7.4 s (surprise 121; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.91, extra_mass_kg=0.501, sag_mm=0.09).
  What I changed: at 7.4 s, expect the extra weight (extra_mass_kg=0.479).
  What I changed: at 7.7 s, expect the extra weight (extra_mass_kg=0.486).
  What I changed: at 9.0 s, expect the extra weight (extra_mass_kg=0.5).
  What I changed: at 9.3 s, expect the extra weight (extra_mass_kg=0.501).
  Ground truth: payload at 0.0 s (mass=0.5011873244080891).

Episode 19: goal on(red, plate): SUCCESS after 78 planning steps (7.8 s).
  Progress: above_object at 1.1 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 4.5 s, over_target at 5.3 s, lowered at 7.1 s, opened at 7.3 s, released at 7.8 s.
  Surprise: my predictions failed at 1 steps, most at 4.5 s (surprise 130; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.87, extra_mass_kg=0.497, sag_mm=0.03).
  What I changed: at 4.5 s, expect the extra weight (extra_mass_kg=0.499).
  What I changed: at 4.8 s, expect the extra weight (extra_mass_kg=0.501).
  Ground truth: payload at 0.0 s (mass=0.494156853472275).

Episode 20: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.3 s, at_object at 13.2 s, grasped at 13.3 s, lifted at 13.8 s, over_target at 14.2 s.
  Surprise: my predictions failed at 7 steps, most at 10.0 s (surprise 1004; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=98, end=101, force_N=[-1.4, 1.2, -14.9], offset_mm=[0.1, -0.3, -0.3]).
  What I changed: at 5.1 s, expect the extra weight (extra_mass_kg=0.472).
  What I changed: at 5.4 s, expect the extra weight (extra_mass_kg=0.474).
  What I changed: at 7.0 s, expect the extra weight (extra_mass_kg=0.48).
  What I changed: at 7.3 s, expect the extra weight (extra_mass_kg=0.484).
  Ground truth: payload at 0.0 s (mass=0.48461553344437613).

Episode 21: goal on(red, plate): SUCCESS after 71 planning steps (7.1 s).
  Progress: above_object at 1.6 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.5 s, over_target at 5.3 s, lowered at 6.5 s, opened at 6.7 s, released at 7.1 s.
  Surprise: my predictions failed at 1 steps, most at 3.5 s (surprise 84; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=4.12, extra_mass_kg=0.42, sag_mm=0.0).
  What I changed: at 3.5 s, expect the extra weight (extra_mass_kg=0.399).
  What I changed: at 3.8 s, expect the extra weight (extra_mass_kg=0.412).
  What I changed: at 5.3 s, expect the extra weight (extra_mass_kg=0.416).
  What I changed: at 5.6 s, expect the extra weight (extra_mass_kg=0.416).
  Ground truth: payload at 0.0 s (mass=0.41510326627856503).

Episode 22: goal on(red, plate): SUCCESS after 106 planning steps (10.6 s).
  Progress: above_object at 1.2 s, at_object at 5.9 s, grasped at 6.0 s, lifted at 6.8 s, over_target at 7.9 s, lowered at 10.1 s, opened at 10.3 s, released at 10.6 s.
  Surprise: my predictions failed at 1 steps, most at 6.8 s (surprise 189; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.89, extra_mass_kg=0.601, sag_mm=0.11).
  What I changed: at 6.8 s, expect the extra weight (extra_mass_kg=0.598).
  What I changed: at 7.1 s, expect the extra weight (extra_mass_kg=0.597).
  What I changed: at 8.4 s, expect the extra weight (extra_mass_kg=0.598).
  What I changed: at 8.7 s, expect the extra weight (extra_mass_kg=0.601).
  Ground truth: payload at 0.0 s (mass=0.5991629807367633).

Episode 23: goal on(red, plate): SUCCESS after 85 planning steps (8.5 s).
  Progress: above_object at 1.3 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.5 s, over_target at 6.7 s, lowered at 7.9 s, opened at 8.1 s, released at 8.5 s.
  Surprise: my predictions failed at 1 steps, most at 3.5 s (surprise 184; spikes start at 32).
  My explanation: the object is heavier than I expected, probability 1.00 (extra_weight_N=5.88, extra_mass_kg=0.599, sag_mm=0.0).
  What I changed: at 3.5 s, expect the extra weight (extra_mass_kg=0.588).
  What I changed: at 3.8 s, expect the extra weight (extra_mass_kg=0.581).
  What I changed: at 5.2 s, expect the extra weight (extra_mass_kg=0.589).
  What I changed: at 5.5 s, expect the extra weight (extra_mass_kg=0.59).
  What I changed: at 6.9 s, expect the extra weight (extra_mass_kg=0.594).
  What I changed: at 7.2 s, expect the extra weight (extra_mass_kg=0.596).
  Ground truth: payload at 0.0 s (mass=0.594250601632869).

Episode 24: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 9.6 s, at_object at 6.6 s, grasped at 6.7 s, lifted at 7.6 s.
  Surprise: my predictions failed at 11 steps, most at 3.5 s (surprise 456; spikes start at 32).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=35, sink_mm=51.3, slide_mm=[1.1, 8.6], weight_lost_N=0.79).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 25: goal on(red, plate): SUCCESS after 88 planning steps (8.8 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 4.8 s, at_object at 6.1 s, grasped at 6.2 s, lifted at 6.7 s, over_target at 7.7 s, lowered at 8.1 s, opened at 8.4 s, released at 8.8 s.
  Surprise: my predictions failed at 1 steps, most at 3.9 s (surprise 214; spikes start at 32).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=39, sink_mm=34.7, slide_mm=[-3.9, 0.3], weight_lost_N=0.73).
  What I changed: at 3.9 s, carry it low, grasped at its centre.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 26: goal on(red, plate): SUCCESS after 149 planning steps (14.9 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 11.6 s, at_object at 13.2 s, grasped at 13.3 s, lifted at 13.9 s, over_target at 14.3 s.
  Surprise: my predictions held throughout.
  My explanation: the object is sliding in my hand, probability 0.65 (onset=98, sink_mm=9.2, slide_mm=[-1.2, -2.4], weight_lost_N=0.77).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 27: goal on(red, plate): SUCCESS after 86 planning steps (8.6 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 6.0 s, at_object at 6.7 s, grasped at 6.8 s, lifted at 7.0 s, over_target at 7.3 s, lowered at 7.9 s, opened at 8.1 s, released at 8.6 s.
  Surprise: my predictions failed at 1 steps, most at 5.3 s (surprise 449; spikes start at 32).
  My explanation: the object is sliding in my hand, probability 1.00 (onset=53, sink_mm=48.3, slide_mm=[1.6, 10.6], weight_lost_N=0.78).
  What I changed: at 5.3 s, carry it low, grasped at its centre.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 28: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 12.0 s, at_object at 13.7 s, grasped at 13.8 s, lifted at 14.5 s.
  Surprise: my predictions failed at 4 steps, most at 7.9 s (surprise 2332; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=79, end=81, force_N=[8.8, 1.3, -49.7], offset_mm=[0.7, 0.4, 1.2]).
  What I changed: at 6.2 s, carry it low, grasped at its centre.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 29: goal on(red, plate): SUCCESS after 137 planning steps (13.7 s).
  Setbacks: I lost the object 1 time(s) and went back to re-grasp it.
  Progress: above_object at 5.7 s, at_object at 6.7 s, grasped at 6.8 s, lifted at 7.3 s, over_target at 8.3 s, lowered at 12.9 s, opened at 13.2 s, released at 13.7 s.
  Surprise: my predictions failed at 18 steps, most at 10.3 s (surprise 1043; spikes start at 32).
  My explanation: something happened that I cannot explain yet, probability 1.00 (residual_rms=2.03).
  What I changed: at 5.2 s, carry it low, grasped at its centre.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 30: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.2 s.
  Surprise: my predictions held throughout.
  My explanation: nothing unusual happened.
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 31: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Setbacks: I lost the object 2 time(s) and went back to re-grasp it.
  Progress: above_object at 11.5 s, at_object at 8.5 s, grasped at 2.7 s, lifted at 3.4 s.
  Surprise: my predictions failed at 8 steps, most at 4.8 s (surprise 994; spikes start at 32).
  My explanation: something pushed my arm, probability 1.00 (onset=48, end=48, force_N=[-4.1, -5.8, -10.4], offset_mm=[1.1, 2.2, 0.1]).
  Ground truth: friction at 0.0 s (scale=0.05).

Episode 32: goal on(red, plate): SUCCESS after 80 planning steps (8.0 s).
  Progress: above_object at 1.3 s, at_object at 2.1 s, grasped at 2.2 s, lifted at 2.9 s, over_target at 4.8 s, lowered at 7.4 s, opened at 7.6 s, released at 8.0 s.
  Surprise: my predictions failed at 1 steps, most at 4.3 s (surprise 1084; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=43, camera_offset_mm=[-19.6, -45.5, -2.2], seen_jump=[0.0196, 0.0455, 0.0022]).
  What I changed: at 4.3 s, recalibrated the camera (correction_mm=[-19.600000381469727, -45.5, -2.200000047683716]).
  Ground truth: camera_shift at 4.3 s (offset=[-0.019509432905790138, -0.04550820000891167, -0.0022215715204179243]).

Episode 33: goal on(red, plate): SUCCESS after 77 planning steps (7.7 s).
  Progress: above_object at 1.2 s, at_object at 3.6 s, grasped at 3.9 s, lifted at 4.8 s, over_target at 5.4 s, lowered at 7.0 s, opened at 7.2 s, released at 7.7 s.
  Surprise: my predictions failed at 1 steps, most at 3.6 s (surprise 769; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=36, camera_offset_mm=[34.7, 38.4, 0.5], seen_jump=[-0.0347, -0.0384, -0.0005]).
  What I changed: at 3.6 s, recalibrated the camera (correction_mm=[34.70000076293945, 38.400001525878906, 0.5]).
  Ground truth: camera_shift at 3.7 s (offset=[0.03412972380982202, 0.038759925793879234, 0.000507086449514518]).

Episode 34: goal on(red, plate): FAILED after 149 planning steps (14.9 s).
  Progress: above_object at 1.0 s.
  Surprise: my predictions failed at 1 steps, most at 3.8 s (surprise 899; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=38, camera_offset_mm=[-16.5, 41.5, 8.7], seen_jump=[0.0165, -0.0415, -0.0087]).
  What I changed: at 3.8 s, recalibrated the camera (correction_mm=[-16.5, 41.5, 8.699999809265137]).
  Ground truth: camera_shift at 3.8 s (offset=[-0.016472141088644972, 0.04141985767907811, 0.008680870319124992]).

Episode 35: goal on(red, plate): SUCCESS after 84 planning steps (8.4 s).
  Progress: above_object at 1.1 s, at_object at 3.7 s, grasped at 3.8 s, lifted at 4.4 s, over_target at 5.3 s, lowered at 7.6 s, opened at 7.8 s, released at 8.4 s.
  Surprise: my predictions failed at 1 steps, most at 4.5 s (surprise 977; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=45, camera_offset_mm=[-29.6, 36.5, -3.3], seen_jump=[0.0296, -0.0365, 0.0033]).
  What I changed: at 4.5 s, recalibrated the camera (correction_mm=[-29.600000381469727, 36.5, -3.299999952316284]).
  Ground truth: camera_shift at 4.5 s (offset=[-0.02954581861406186, 0.0367393523406594, -0.0035626121784811575]).

Episode 36: goal on(red, plate): SUCCESS after 86 planning steps (8.6 s).
  Progress: above_object at 1.2 s, at_object at 4.0 s, grasped at 4.1 s, lifted at 4.8 s, over_target at 6.5 s, lowered at 7.9 s, opened at 8.1 s, released at 8.6 s.
  Surprise: my predictions failed at 1 steps, most at 3.9 s (surprise 723; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=39, camera_offset_mm=[-33.4, -22.3, 7.9], seen_jump=[0.0334, 0.0223, -0.0079]).
  What I changed: at 3.9 s, recalibrated the camera (correction_mm=[-33.400001525878906, -22.299999237060547, 7.900000095367432]).
  Ground truth: camera_shift at 4.0 s (offset=[-0.03329571227517061, -0.022414311202739488, 0.007805487040095848]).

Episode 37: goal on(red, plate): SUCCESS after 75 planning steps (7.5 s).
  Progress: above_object at 1.6 s, at_object at 2.5 s, grasped at 2.6 s, lifted at 3.4 s, over_target at 5.2 s, lowered at 6.8 s, opened at 7.0 s, released at 7.5 s.
  Surprise: my predictions failed at 1 steps, most at 2.4 s (surprise 910; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=24, camera_offset_mm=[7.2, 48.0, -8.3], seen_jump=[-0.0072, -0.048, 0.0083]).
  What I changed: at 2.4 s, recalibrated the camera (correction_mm=[7.199999809265137, 48.0, -8.300000190734863]).
  Ground truth: camera_shift at 2.4 s (offset=[0.00696497483556975, 0.04819493730202898, -0.008319693128352304]).

Episode 38: goal on(red, plate): SUCCESS after 148 planning steps (14.8 s).
  Progress: above_object at 1.2 s, at_object at 10.9 s, grasped at 11.0 s, lifted at 11.7 s, over_target at 12.9 s, lowered at 14.1 s, opened at 14.3 s, released at 14.8 s.
  Surprise: my predictions failed at 1 steps, most at 2.1 s (surprise 1239; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=21, camera_offset_mm=[26.7, -46.6, 7.6], seen_jump=[-0.0267, 0.0466, -0.0076]).
  What I changed: at 2.1 s, recalibrated the camera (correction_mm=[26.700000762939453, -46.599998474121094, 7.599999904632568]).
  Ground truth: camera_shift at 2.1 s (offset=[0.02660516818779524, -0.046545819901094805, 0.007529684616214076]).

Episode 39: goal on(red, plate): SUCCESS after 76 planning steps (7.6 s).
  Progress: above_object at 1.3 s, at_object at 2.6 s, grasped at 2.7 s, lifted at 3.4 s, over_target at 5.6 s, lowered at 6.9 s, opened at 7.1 s, released at 7.6 s.
  Surprise: my predictions failed at 1 steps, most at 1.8 s (surprise 620; spikes start at 32).
  My explanation: my camera moved: everything I see jumped while my hand did not, probability 1.00 (onset=18, camera_offset_mm=[37.5, 14.4, -7.0], seen_jump=[-0.0375, -0.0144, 0.007]).
  What I changed: at 1.8 s, recalibrated the camera (correction_mm=[37.5, 14.399999618530273, -7.0]).
  Ground truth: camera_shift at 1.8 s (offset=[0.037399958222325565, 0.0144198143019028, -0.006994410662103219]).
```
