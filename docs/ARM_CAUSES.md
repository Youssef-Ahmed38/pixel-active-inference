# Stage 1: learned arm cause likelihoods

Frozen final diagnosis: **97/100 vs 87/100** with exact-state evidence, and
**78/100 vs 66/100** with camera evidence, against the old method on the same
episodes. Paired layout-bootstrap accuracy gains are 10 percentage points
[3, 16] and 12 points [2, 22], respectively. These are improvements, not perfect
reliability: camera slip recall is still 65%, and physical timing/offset errors
remain. Full intervals, failures and controller results are below.

This study learns the likelihood of the arm's own evidence under `none`, `push`,
`heavier_object`, `slippery_object`, and `camera_shift`. The old analytic inference
module remains unchanged as the baseline and as a physical parameter estimator.
The final results below compare diagnoses on **exactly the same recorded sensor
histories**. A separate controller comparison measures whether learned online
reports help the task; terminal diagnosis alone cannot establish that.

## Evidence and learning

`pai/causes/learned.py` accepts `StepEvidence` histories: world-model prediction
residuals, their calibrated uncertainty, actions, whether the hand is holding an
object, the current task phase, and time. Neither inference nor feature extraction
accepts an event log, physical disturbance, schedule, or label. Collection captures
the monitor before attaching truth in a separate `scoring` field. The audit replays
both methods, then changes labels and logs and verifies identical learned answers.

Given feature summaries include signed peaks and RMS errors in the twelve
hand/force/object/background channels; contact-conditioned force and relative
object/hand errors; and common versus independent visual displacement. Optional
features include all six old analytic posterior values and physical fit readouts.
These summaries and the old physical equations are authored, not learned. The raw
feature model does not use the old posterior to choose its cause.

What is learned from labelled development episodes: an asinh scale, pooled
centering/scaling, and a mean and shrunk covariance for each cause. For Student-t,
that shrunk moment covariance is used as the density's scale matrix; this is not
iterative Student-t maximum-likelihood fitting. Cross-validation
selects raw / analytic-fit / combined features, covariance shrinkage 0.5 / 0.9 / 1,
Gaussian / Student-t (four degrees of freedom) likelihoods, and uniform / old
class priors. All five conditions from a layout seed are held out together.
Selection maximizes macro recall, breaking ties by fewer false pushes in clean,
heavy and slippery cases. This is model selection on development data, not a
nested or final estimate. No neural network or world model is retrained.

The distribution families, shrinkage grid, covariance ridge 0.1, feature scale
floor 0.05, and priors are given. The broad `unknown` likelihood is a Student-t
with three degrees of freedom, scale 4 in the learned pooled coordinates, and
prior 0.02. Its location/scale coordinates derive from development evidence; its
tail width and prior are authored. There are no labelled real unknown episodes in
this benchmark: synthetic outliers test the mechanism, not real unknown recall.

Terminal and online models are fitted separately. Prefix training uses every
tenth transition, actual alarm prefixes, and the last prefix, with total weight
one per episode. A prefix is labelled `none` until its injection actually starts;
labels are teaching targets only. Input features stop at that prefix. The existing
calibrated surprise trigger and adaptation threshold 0.9 remain given. Missing
physical readouts withhold an adaptation without altering the diagnosis.

Parameter estimates are still the old own-evidence least-squares/onset fits, not
supervised regressions. A heavy diagnosis whose peak window has no load estimate
tries a window around the strongest **sensed held-object** residual. Missing
estimates remain missing. Scoring only credits estimates when the cause is right:
mass absolute error tolerance 0.1 kg, camera offset vector error tolerance 10 mm,
and onset error tolerance 0.3 s. Push and camera onset are matched only to their
own cause's event; slip onset is compared with the first red-object slip, not with
the earlier friction change or another object's slip.

## Integration and comparison limits

The learned monitor subclasses the original monitor. A scoped, process-local
adapter supplies it to the unchanged arm agent. `scripts/arm_causes.py collect
--controller learned --model ...` uses learned prefix reports in actual control
and a learned terminal verdict. The baseline controller is the default. The
adapter restores the original class even on exceptions; concurrent episodes in
one process are unsupported. Other stages and default production behavior have
not been changed. Language will need its caller to opt into the adapter before
its explanations automatically benefit.

Both controllers use the current `configs/tabletop.yaml` (v8 low carry and settle
monitoring), seeded planner, and the existing frozen world model. These are fresh
paired comparisons, not replays of the older v7 controller. Each layout is tested
under five scheduled conditions. If a push never starts before an episode ends,
its actual cause is `none`; its scheduled condition is still retained.

`pixel_slots` uses frozen DINOv2/slot readouts, `refine=False`, with batch-one
perception. This retains the existing given hand proprioception, touch/held-object
and occlusion/permanence rules and known object identities. It is not end-to-end
unsupervised pixel learning. Local cached DINO code/weights are required; this
script performs no downloads. `bump_follows=True` makes the physical camera bump
reach the camera that supplies perception. The historical `pixel_v3` camera bump
was invisible to that camera. Consequently its camera condition is not an
identical historical comparator. Also, historical **35% slippery performance was
task success, while its cause recall was 75%**; the two metrics must not be confused.
The slot-only adapter also has `allow_recalibration=False`: recognizing a bump
does not enable a camera correction in this regime. That existing behavior is
kept for both controllers, and limits camera-shift task success.

Wilson intervals for episode accuracy and recall are descriptive binomial 95%
intervals. Conditions share layout seeds; paired accuracy-gain intervals instead
resample whole layouts (5,000 draws, seed 870000). Controller success differences
use layout resampling with seed 870001. Controller pairs share initial layouts and
disturbance schedules but may follow different trajectories; diagnostic pairs
share the full factual trace. The online replay conditions on actual trigger
times and is not a counterfactual closed-loop success experiment.

## Seeds, provenance, and reproduction

Development artifacts are retained, including the first Gaussian candidate that
lost clean-episode recall. Initial exact development was 75 episodes; expanding
to 150 and allowing a shared heavy-tailed family selected a different model.
Old artifacts are not overwritten. Final seeds and frozen model/source hashes are
registered in `results/arm_causes_plan.json` before final collection. The seed
inventory searches `slice_v*`, `pixel_v*`, `language_causes_*`, and `arm_causes_*`.
Historical slice files omit explicit episode seeds, so the inventory cannot prove
their absence alone; the old script defaults (1000 evaluation, 5000 calibration)
and documented language ranges are also excluded by choosing the large ranges here.

Every collection checkpoints per-episode own traces and scoring records in gzip
JSON, plus commands, configuration, checkpoint hashes and source snapshots. A
failed collection retains its interruption and partial evidence; `--resume`
requires matching identities and skips completed cases. JSON traces retain array
dtypes because analytic fits use float32 residual arithmetic. Early development
traces omitted dtype fields; their residuals/actions were verified exactly
float32-representable, and the decoder restores those original dtypes.

Run from the repository root, using the project interpreter:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONHASHSEED='0'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$py='C:\projects\pixel-active-inference\.venv\Scripts\python.exe'
& $py -c "import pai; print(pai.__file__)"
& $py scripts/arm_causes.py calibrate --seed 819000 --count 8 --output results/arm_causes_exact_calibration.json.gz
foreach ($s in 820000,821000,822000,823000,824000,825000) {
  & $py scripts/arm_causes.py collect --seed $s --count 5 --calibration results/arm_causes_exact_calibration.json.gz --output "results/arm_causes_exact_dev_$s.json.gz"
}
& $py scripts/arm_causes.py calibrate --regime pixel_slots --seed 828000 --count 8 --output results/arm_causes_pixel_calibration.json.gz
foreach ($s in 830000,831000,832000) {
  & $py scripts/arm_causes.py collect --regime pixel_slots --seed $s --count 5 --calibration results/arm_causes_pixel_calibration.json.gz --output "results/arm_causes_pixel_dev_$s.json.gz"
}
```

Existing outputs are protected from overwrite. Reproduction should use new output
paths or an isolated copy of the repository. GPU collection runs one process at
a time; only CPU fitting runs alongside it. Exact fit/final commands and measured
results follow below.

```powershell
$exactInputs = 820000,821000,822000,823000,824000,825000 | ForEach-Object { "results/arm_causes_exact_dev_$_.json.gz" }
$pixelInputs = 830000,831000,832000 | ForEach-Object { "results/arm_causes_pixel_dev_$_.json.gz" }
& $py scripts/arm_causes.py fit --regime exact --calibration results/arm_causes_exact_calibration.json.gz --inputs $exactInputs --output results/arm_causes_exact_model_frozen.json.gz
& $py scripts/arm_causes.py fit --regime pixel_slots --calibration results/arm_causes_pixel_calibration.json.gz --inputs $pixelInputs --output results/arm_causes_pixel_model_frozen.json.gz
& $py scripts/arm_causes.py collect --controller learned --seed 826000 --count 1 --calibration results/arm_causes_exact_calibration.json.gz --model results/arm_causes_exact_model_frozen.json.gz --output results/arm_causes_exact_controller_development.json.gz
& $py scripts/arm_causes.py collect --regime pixel_slots --controller learned --seed 836000 --count 1 --calibration results/arm_causes_pixel_calibration.json.gz --model results/arm_causes_pixel_model_frozen.json.gz --output results/arm_causes_pixel_controller_development.json.gz
& $py scripts/arm_causes_tests.py --basetemp data/pytest_arm_causes_full_20261028b --output results/arm_causes_tests.json
& $py scripts/arm_causes_study.py register
& $py scripts/arm_causes_study.py run
& $py scripts/arm_causes_report.py
```

The study driver executes the exact argument lists stored in the plan. For each
regime it runs `collect` (baseline controller, 20 layouts), `score`, truth-boundary
`audit`, then `collect --controller learned` (first five layouts), `score` and
`audit`, followed by prefix replay and the paired controller comparison. All
collection commands use disturbance seed 840000, deterministically combined with
layout seed and condition index. Seeds 817000–817001 / 818000 (exact smoke),
829000–829001 / 829100 (camera smoke), and 826000 / 836000 (live-controller smoke)
are development only. Clean calibration uses 819000–819007 exact and
828000–828007 camera. Exact development uses six five-layout blocks at
820000, 821000, 822000, 823000, 824000, 825000; camera development uses three at
830000, 831000, 832000. Final uses 850000–850019 exact and 860000–860019 camera.

The test runner executes `tests/test_causes*.py`, `tests/test_slice.py`,
`tests/test_cognition.py`, and `tests/test_language*.py`. On this managed Windows
machine `mkdir(mode=0700)` makes pytest's temporary directories inaccessible. The
runner uses a **new** directory inside `data` with inherited permissions, without
changing existing ACLs or deleting directories. Use a new basetemp name on reruns.
The resumed regression command passed **93 tests**:

```powershell
& $py scripts/arm_causes_tests.py --basetemp data/pytest_arm_causes_resume_20261028 --output results/arm_causes_tests_resumed.json
```

The original session was interrupted during camera final collection with 32
completed episodes. Running `scripts/arm_causes_study.py run` again retained all
completed exact-state artifacts and resumed camera `collect` with the same
registered arguments plus `--resume`. The first missing case was seed 860006,
heavier object. `results/arm_causes_resume_checkpoint.json` records hashes of the
retained camera prefix and completed exact-state files; the completion audit
checks these rather than replacing completed cases. No fitting or model
selection occurred after registration.
`results/arm_causes_resume_audit.json` confirms that all 32 retained camera
records and eight completed exact-state artifacts are unchanged. Existing world
model, slot model and cached DINOv2 weight hashes are also recorded in
`results/arm_causes_fixed_assets.json`; none was retrained or replaced.

## Known limits

The final gate is deliberately not a claim of 100% reliability. In exact state,
the learned method still mistakes one clean episode for push and one camera
shift for none; the latter answer is highly confident, so density posteriors
must not be described as calibrated correctness probabilities. Improving overall
accuracy does not guarantee improving every cause's recall. No final failures
were used to refit or select a different model.

Slip classification and slip **timing** are different problems: the reused
analytic onset readout is absent in many correctly diagnosed slippery episodes
and often finds a late residual rather than the first physical slip. The tables
retain its poor coverage and error. Heavy mass and camera-offset readouts remain
analytic estimates, with error reported only where available and correctly named.

Only one tabletop goal and singly injected known disturbances are measured.
Multiple simultaneous causes, new disturbance families, calibrated abstention on
real unknowns, transfer to other goals, and integration into default arm/language
callers are not established. The learned monitor is opt-in; the existing baseline
and other stages remain available unchanged. The camera regime inherits grasp,
perception and recalibration limits of the fixed slot-only controller. A better
terminal explanation by itself cannot repair a task that already failed.

Camera final failures remain substantial: 22/100 diagnoses are wrong, including
7/20 slippery cases. Camera-shift recall is slightly worse than the old method
(15/20 vs 16/20); on correctly named camera cases the learned method's reused
offset estimator has mean vector error 66.8 mm, far outside the 10 mm tolerance.
The camera heavy-mass MAE is 0.0426 kg with 19/19 estimate coverage and 17/19
within 0.1 kg, but its mean includes a different set of correctly diagnosed
episodes than the baseline. Matched-case comparisons below avoid implying that
the estimator itself improved just because classification changed.

## Frozen final results

### exact: seeds 850000–850019

| Diagnosis | Old analytic | Learned |
|---|---:|---:|
| Overall | 87/100 = 87.0% [79.0, 92.2] | 97/100 = 97.0% [91.5, 99.0] |
| none recall | 20/20 = 100.0% [83.9, 100.0] | 19/20 = 95.0% [76.4, 99.1] |
| push recall | 20/20 = 100.0% [83.9, 100.0] | 19/20 = 95.0% [76.4, 99.1] |
| heavier_object recall | 18/20 = 90.0% [69.9, 97.2] | 20/20 = 100.0% [83.9, 100.0] |
| slippery_object recall | 9/20 = 45.0% [25.8, 65.8] | 20/20 = 100.0% [83.9, 100.0] |
| camera_shift recall | 20/20 = 100.0% [83.9, 100.0] | 19/20 = 95.0% [76.4, 99.1] |

Balanced accuracy: old 87.0%, learned 97.0%.
Paired gain: 10.0 percentage points; layout-bootstrap 95% interval [3.0, 16.0]. New-only correct 13; old-only correct 3.
Unknown predictions: {'old': 0, 'new': 0}. Scheduled disturbances that never fired: 0.
False pushes by true cause: `{'old': {'none': 0, 'heavier_object': 2, 'slippery_object': 11}, 'new': {'none': 1, 'heavier_object': 0, 'slippery_object': 0}}`.

Confusion rows: none, push, heavier, slippery, camera. Columns: those five, unknown.

```text
old [[20, 0, 0, 0, 0, 0], [0, 20, 0, 0, 0, 0], [0, 2, 18, 0, 0, 0], [0, 11, 0, 9, 0, 0], [0, 0, 0, 0, 20, 0]]
new [[19, 1, 0, 0, 0, 0], [0, 19, 0, 1, 0, 0], [0, 0, 20, 0, 0, 0], [0, 0, 0, 20, 0, 0], [1, 0, 0, 0, 19, 0]]
```

| Parameter, correct cause only | Old coverage / MAE | Learned coverage / MAE |
|---|---:|---:|
| heavier_object: mass_abs_error_kg | 18/18 / 0.0024 kg; 18 within tolerance | 20/20 / 0.0050 kg; 20 within tolerance |
| camera_shift: offset_error_mm | 20/20 / 0.1074 mm; 20 within tolerance | 19/19 / 0.1023 mm; 19 within tolerance |
| push: onset_error_s | 20/20 / 0.0260 s; 20 within tolerance | 19/19 / 0.0274 s; 19 within tolerance |
| slippery_object: onset_error_s | 6/9 / 1.6900 s; 3 within tolerance | 10/20 / 2.9380 s; 3 within tolerance |
| camera_shift: onset_error_s | 20/20 / 0.0440 s; 20 within tolerance | 19/19 / 0.0432 s; 19 within tolerance |

Coverage denominators are correctly diagnosed cases, not all injected cases. Errors across methods can involve different cases. Matched-case MAEs: `{'mass_abs_error_kg': {'n': 18, 'old': 0.002378713292107312, 'new': 0.002378713292107312}, 'offset_error_mm': {'n': 19, 'old': 0.10231147981606911, 'new': 0.10231147981606911}, 'onset_error_s': {'n': 44, 'old': 0.26090909090909087, 'new': 0.26090909090909087}}`.

#### Actual closed-loop controller subset

Seeds 850000–850004, frozen prefix models; task goal success:

| Condition | Analytic controller | Learned controller |
|---|---:|---:|
| none | 5/5 = 100.0% [56.6, 100.0] | 5/5 = 100.0% [56.6, 100.0] |
| push | 5/5 = 100.0% [56.6, 100.0] | 5/5 = 100.0% [56.6, 100.0] |
| heavier_object | 5/5 = 100.0% [56.6, 100.0] | 5/5 = 100.0% [56.6, 100.0] |
| slippery_object | 5/5 = 100.0% [56.6, 100.0] | 5/5 = 100.0% [56.6, 100.0] |
| camera_shift | 4/5 = 80.0% [37.6, 96.4] | 5/5 = 100.0% [56.6, 100.0] |
| all | 24/25 = 96.0% [80.5, 99.3] | 25/25 = 100.0% [86.7, 100.0] |

Task-success difference layout-bootstrap interval: [0.0, 12.0] points. Learned-only successes 1; baseline-only successes 0.
This five-layout controller subset is exploratory. A zero-width bootstrap interval means all observed paired differences were zero; it does not establish zero effect on unseen layouts.
Terminal diagnosis on the learned controller's own trajectories: learned 24/25 = 96.0% [80.5, 99.3]; analytic post-hoc comparator 20/25 = 80.0% [60.9, 91.1]. This is separate from the main baseline-trajectory table.

#### Held-out evidence examples and failures

| Seed / scheduled condition | Event-log cause (scoring only) | Old | Learned (confidence) | Goal |
|---|---|---|---|---:|
| 850003 / push | push | push | slippery_object (0.569) | True |
| 850003 / camera_shift | camera_shift | camera_shift | none (1.000) | False |
| 850017 / none | none | none | push (0.941) | True |
| 850000 / slippery_object | slippery_object | push | slippery_object (1.000) | True |
| 850001 / slippery_object | slippery_object | push | slippery_object (1.000) | True |
| 850003 / heavier_object | heavier_object | push | heavier_object (1.000) | True |
| 850004 / slippery_object | slippery_object | push | slippery_object (1.000) | True |
| 850005 / slippery_object | slippery_object | push | slippery_object (1.000) | False |
| 850000 / none | none | none | none (1.000) | True |
| 850000 / push | push | push | push (1.000) | True |
| 850000 / heavier_object | heavier_object | heavier_object | heavier_object (1.000) | True |

Development selection (layout-held-out; not final):

| Model | Features / shrinkage / family / prior | Accuracy | Balanced accuracy |
|---|---|---:|---:|
| terminal | raw / 0.9 / student_t4 / uniform | 94.0% | 94.0% |
| online | fused / 0.5 / student_t4 / uniform | 82.0% | 83.9% |

Terminal / prefix training records: {'terminal': 150, 'online_prefixes': 2290}.

| Prefix replay on final factual trace | Old accuracy | Learned accuracy |
|---|---:|---:|
| at_alarms | 220/466 | 403/466 |
| last_report_per_episode | 67/100 | 96/100 |

### pixel: seeds 860000–860019

| Diagnosis | Old analytic | Learned |
|---|---:|---:|
| Overall | 66/100 = 66.0% [56.3, 74.5] | 78/100 = 78.0% [68.9, 85.0] |
| none recall | 12/20 = 60.0% [38.7, 78.1] | 15/20 = 75.0% [53.1, 88.8] |
| push recall | 16/20 = 80.0% [58.4, 91.9] | 16/20 = 80.0% [58.4, 91.9] |
| heavier_object recall | 13/20 = 65.0% [43.3, 81.9] | 19/20 = 95.0% [76.4, 99.1] |
| slippery_object recall | 9/20 = 45.0% [25.8, 65.8] | 13/20 = 65.0% [43.3, 81.9] |
| camera_shift recall | 16/20 = 80.0% [58.4, 91.9] | 15/20 = 75.0% [53.1, 88.8] |

Balanced accuracy: old 66.0%, learned 78.0%.
Paired gain: 12.0 percentage points; layout-bootstrap 95% interval [2.0, 22.0]. New-only correct 18; old-only correct 6.
Unknown predictions: {'old': 20, 'new': 0}. Scheduled disturbances that never fired: 0.
False pushes by true cause: `{'old': {'none': 1, 'heavier_object': 1, 'slippery_object': 2}, 'new': {'none': 1, 'heavier_object': 0, 'slippery_object': 0}}`.

Confusion rows: none, push, heavier, slippery, camera. Columns: those five, unknown.

```text
old [[12, 1, 0, 0, 3, 4], [0, 16, 0, 2, 0, 2], [1, 1, 13, 0, 1, 4], [0, 2, 0, 9, 3, 6], [0, 0, 0, 0, 16, 4]]
new [[15, 1, 0, 1, 3, 0], [1, 16, 0, 2, 1, 0], [0, 0, 19, 1, 0, 0], [2, 0, 0, 13, 5, 0], [4, 0, 0, 1, 15, 0]]
```

| Parameter, correct cause only | Old coverage / MAE | Learned coverage / MAE |
|---|---:|---:|
| heavier_object: mass_abs_error_kg | 13/13 / 0.0270 kg; 12 within tolerance | 19/19 / 0.0426 kg; 17 within tolerance |
| camera_shift: offset_error_mm | 16/16 / 44.3535 mm; 6 within tolerance | 15/15 / 66.7720 mm; 6 within tolerance |
| push: onset_error_s | 16/16 / 0.0400 s; 16 within tolerance | 16/16 / 0.0400 s; 16 within tolerance |
| slippery_object: onset_error_s | 3/9 / 4.4733 s; 1 within tolerance | 4/13 / 3.3750 s; 2 within tolerance |
| camera_shift: onset_error_s | 16/16 / 1.2087 s; 6 within tolerance | 15/15 / 1.3667 s; 8 within tolerance |

Coverage denominators are correctly diagnosed cases, not all injected cases. Errors across methods can involve different cases. Matched-case MAEs: `{'mass_abs_error_kg': {'n': 13, 'old': 0.026993381062160475, 'new': 0.026993381062160475}, 'offset_error_mm': {'n': 11, 'old': 16.97981261879548, 'new': 16.97981261879548}, 'onset_error_s': {'n': 30, 'old': 0.8113333333333334, 'new': 0.8113333333333334}}`.

#### Actual closed-loop controller subset

Seeds 860000–860004, frozen prefix models; task goal success:

| Condition | Analytic controller | Learned controller |
|---|---:|---:|
| none | 4/5 = 80.0% [37.6, 96.4] | 4/5 = 80.0% [37.6, 96.4] |
| push | 4/5 = 80.0% [37.6, 96.4] | 4/5 = 80.0% [37.6, 96.4] |
| heavier_object | 5/5 = 100.0% [56.6, 100.0] | 5/5 = 100.0% [56.6, 100.0] |
| slippery_object | 3/5 = 60.0% [23.1, 88.2] | 3/5 = 60.0% [23.1, 88.2] |
| camera_shift | 1/5 = 20.0% [3.6, 62.4] | 1/5 = 20.0% [3.6, 62.4] |
| all | 17/25 = 68.0% [48.4, 82.8] | 17/25 = 68.0% [48.4, 82.8] |

Task-success difference layout-bootstrap interval: [0.0, 0.0] points. Learned-only successes 0; baseline-only successes 0.
This five-layout controller subset is exploratory. A zero-width bootstrap interval means all observed paired differences were zero; it does not establish zero effect on unseen layouts.
Terminal diagnosis on the learned controller's own trajectories: learned 19/25 = 76.0% [56.6, 88.5]; analytic post-hoc comparator 14/25 = 56.0% [37.1, 73.3]. This is separate from the main baseline-trajectory table.

#### Held-out evidence examples and failures

| Seed / scheduled condition | Event-log cause (scoring only) | Old | Learned (confidence) | Goal |
|---|---|---|---|---:|
| 860000 / none | none | push | push (0.853) | False |
| 860000 / push | push | unknown | camera_shift (0.853) | True |
| 860001 / slippery_object | slippery_object | push | camera_shift (0.996) | True |
| 860003 / none | none | unknown | camera_shift (1.000) | True |
| 860003 / push | push | slippery_object | slippery_object (1.000) | False |
| 860000 / camera_shift | camera_shift | unknown | camera_shift (1.000) | False |
| 860001 / heavier_object | heavier_object | none | heavier_object (1.000) | True |
| 860003 / heavier_object | heavier_object | push | heavier_object (1.000) | True |
| 860003 / slippery_object | slippery_object | unknown | slippery_object (1.000) | False |
| 860003 / camera_shift | camera_shift | unknown | camera_shift (1.000) | False |
| 860000 / heavier_object | heavier_object | heavier_object | heavier_object (1.000) | True |
| 860000 / slippery_object | slippery_object | slippery_object | slippery_object (1.000) | True |
| 860001 / none | none | none | none (1.000) | True |

Development selection (layout-held-out; not final):

| Model | Features / shrinkage / family / prior | Accuracy | Balanced accuracy |
|---|---|---:|---:|
| terminal | fused / 0.9 / student_t4 / uniform | 78.7% | 78.7% |
| online | fused / 0.5 / student_t4 / legacy | 62.9% | 64.1% |

Terminal / prefix training records: {'terminal': 75, 'online_prefixes': 1242}.

| Prefix replay on final factual trace | Old accuracy | Learned accuracy |
|---|---:|---:|
| at_alarms | 176/472 | 330/472 |
| last_report_per_episode | 56/100 | 81/100 |

