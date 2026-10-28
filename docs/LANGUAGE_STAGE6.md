# Stage 6: grounded text and physical instructions

The runtime induces a lexicon by intersecting semantic concepts across teaching
examples, then induces ordered sentence templates. The teacher alone owns the
English/Arabic dictionaries. Runtime inference loads portable JSON checkpoints.
This is controlled template language, not fluent conversation or audio recognition.

## Architecture

`GroundedLanguage` maps text to `GroundedFrame`, realizes frames as text, and
resolves entity descriptions into shared `GoalRelation` objects.
`LanguageSession` executes those goals through the existing `CognitiveController`,
supports ordered instructions, answers observed relations, and narrates relation
gains and explicit skill stalls. Unknown relations abstain; a negative answer
requires an explicitly observed false relation. Cause posterior scores alone do
not become factual explanations. A success flag alone cannot create narration.

`LearnedAttributes` learns kind/color prototypes from labeled entity-token
sensations. `GroundedArmAdapter` enriches the existing arm belief using these
prototypes. It does not consult object names to infer attributes at runtime.
The underlying perception still uses privileged entity tokens, independent of
Jobs A/C. The controlled benchmark colors the plate blue and bowl green, keeping
geometry and physical properties unchanged. Containment uses the main project's
`in` predicate, expressed as “inside” in English and “داخل” in Arabic.

## Verified

- Each language: 1,848 teaching examples; 456/456 held-out parses and realization
  roundtrips; 84/84 instruction goal resolutions. Entire source-color/predicate/
  target-kind combinations are held out across intents.
- MuJoCo arm sequences on seeds 123 and 124, English and Arabic: **4/4 sequences**,
  **8/8 final goals satisfied**. Each sequence places the red block on the blue
  plate, then the yellow block inside the green bowl. Both combinations are held
  out from template teaching. Runs took 148 and 117 actions respectively, identically
  in both languages under paired planner RNG.
- Every target action was narrated from an observed relation gain, and final
  scene questions returned the learned affirmative answer.
- Unknown words, ambiguous entities, failed commands, false/unknown questions,
  checkpoint persistence and sequence preflight have regression tests.

Reports: `results/language_stage6.json`, `results/language_arm.json`.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts/language_stage6.py
.venv\Scripts\python.exe scripts/language_arm.py --seeds 123 124
.venv\Scripts\python.exe -m pytest tests/test_language.py -q
```

Physical execution requires the existing arm assets and world-model checkpoint.
The controller uses CPU MPPI; language work does not require Jobs A/C.

## Recorded-scene conversations

`language_arm.py` now saves complete shared cognitive episodes to JSONL.
`conversations.py` turns observed relations, explicit stalls and intended goals
into teacher conversations, with a source episode and belief/outcome index on
each utterance. The audit checks the exact semantic frame against that source,
including entity attributes. A success flag alone cannot create a completed
action statement, and a posterior cause guess cannot create a causal fact.

Four additional physical layouts, seeds 130–133, completed both goal relations.
Sequences took 142, 129, 218 and 208 actions. Teaching uses layouts 130–131;
evaluation uses 132–133. In each language, 94 recorded-scene teaching rows were
added to the compositional curriculum; **118/118** held-out scene rows parsed
exactly, roundtripped and matched their supporting observations.
The scene split tests new layouts with shared templates and vocabulary; it does
not establish novel grammar learning or noisy perception robustness.

```powershell
.venv\Scripts\python.exe scripts/language_arm.py --languages english --seeds 130 131 132 133 --output results/language_arm_conversation_collection.json --episodes runs/language/conversation_episodes.jsonl
.venv\Scripts\python.exe scripts/language_conversations.py --episodes runs/language/conversation_episodes.jsonl
```

Reports: `results/language_arm_conversation_collection.json`,
`results/language_conversations.json`. All generated teacher utterances retain
their provenance in `runs/language/conversations_english.jsonl` and the Arabic
counterpart. English physical logs can teach both languages because the stored
meanings and observations are independent of the displayed utterance.

## Explicit causal hypotheses

`LanguageSession.narrate_hypotheses` can describe an episode-level cause guess
using a distinct learned hypothesis intent: English "possibly" and Arabic
"ربما". It requires an observed stall, its corresponding subsequent belief,
a finite normalized posterior, a unique most probable cause and a confidence
threshold. The source indices and confidence are retained; `cause_verified` is
always false. Ordinary factual narration still contains no inferred causes.

Separate models in `runs/language/hypotheses` preserve the existing language
checkpoints. The expanded curriculum passed **483/483** held-out parses and
realization roundtrips in each language, including **27/27** hypothesis sentences.
The eight recorded arm episodes yielded zero cause hypotheses and zero verified
causes: they contain no suitable cause posterior. No causal evidence was invented.

```powershell
.venv\Scripts\python.exe scripts/language_hypotheses.py
```

Report: `results/language_hypotheses.json`. The combined language suite passed
**27 tests**, including abstention, missing observations and invalid probabilities.

## Remaining causal and speech work

Broad disturbed physical tasks, observed causal explanations, spoken audio and open conversation are unbuilt.
Current causal sentences are teacher templates, not a verified explanation gate.

## Evidence-grounded causal language (October 2026)

This addition connects bilingual reports to the existing arm cause inference and
local counterfactual credit model. The historical results above remain as an
archive: physical language results produced before perceived relations replaced
simulator predicates must not be cited as current leak-free scores. The new
evaluation below independently checks a fresh disturbed placement task; it does
not revalidate every historical instruction sequence or conversation corpus.

### Design and evidence boundary

`pai/language/causal.py` accepts an `EpisodeEvidence` allowlist: cause posterior,
fitted parameters, recorded adaptations, loss-of-hold fallbacks, a perceived final
goal observation, and local `CreditResult` records. Its conversion from an arm
record never reads `success`, `events`, or `true_disturbances`. A regression test
raises on any attempt to access a non-allowlisted record field. The runtime module
imports neither simulator nor teacher code.

The slice agent's record still stores simulator-scored success. The evaluation
therefore explicitly discards that flag when constructing language evidence. It
takes a final observation after settling through `CameraFramePerception`, using
the agent's accumulated camera correction, and computes `perceived_predicates`
from those tokens. Simulator success and event logs enter only the scoring/output
section after both languages have produced their utterances. This remains the
declared privileged entity-token interface, including supervised kind/color
features and known object geometry; it is not pixel perception or Jobs A/C.

A cause is reported only for a finite normalized posterior with a unique maximum
of at least **0.9**, a known cause other than `unknown`, and a taught phrase. All
accepted causes are explicitly hedged as the agent's interpretation. Missing,
ambiguous, malformed, low-confidence, and unknown evidence abstains. No report
from the surprise monitor means insufficient evidence, not proof of no disturbance.
Finite estimates retain their units: extra mass in kg, camera translation as a
signed world-coordinate x/y/z vector in mm, onset in planning steps, and slip sink
in mm. Recorded adaptations may be reported even if the final cause is uncertain.
The 0.9 threshold is a fixed reporting policy, not an empirically calibrated
probability of correctness; approximate Bayesian model comparison can still be
confidently wrong under model mismatch.

### What is given and learned

The teacher owns English and Arabic dictionaries in `causal_teacher.py`. The
semantic clause keys, numeric slot bindings, cause labels for development
episodes, confidence threshold, report assembly policy, and question intents are
given. English and Arabic `ClauseLanguage` instances are fitted separately; no
English checkpoint or translation routine is used to generate Arabic at runtime.
Cause phrases are acquired from teacher-labelled development episode/observation
pairs. A cause absent from teaching is not available for realization. Numeric and
episode slots are induced by replacing the paired annotated values and requiring
consistent templates; checkpoints are stored as JSON inside the result file.
The existing independently fitted `GroundedLanguage` models parse arm instructions
and resolve the taught attributes into the shared relational goal.

This is deliberately limited **supervised phrase/template induction**. Whole cause
phrases can be memorized; it does not establish word discovery from competing
scene referents, novel grammar, free conversation, or the full cross-situational
learning proposed in the independent audit. Development episodes teach labels;
final truth never teaches or selects a phrase.

### Questions and counterfactuals

Four question families are evaluated on new episodes in each language: episode
explanation, cause of a drop, removal of a push, and opening the fingers early.
The final numbered questions contain episode numbers absent from teaching (which
uses 11 and 22). This tests slot composition within taught question grammar,
not unseen paraphrases. Unknown utterances and references to another episode
abstain.

The drop question always abstains: a loss-of-hold fallback and an episode-level
cause posterior do not identify the cause of a specific drop. The existing credit
model cannot remove an external push, so that question also abstains. Opening the
fingers early is answered only from one finite, valid `open_early` result returned
by `pai.causes.credit.assign_phase_credit`. The answer describes a **local model
prediction about the tested subgoal**, not the whole task or a verified alternate
physical outcome. Invalid baselines, unavailable interventions, duplicate results,
and nonfinite costs abstain. Model-consistency and abstention-policy checks are
reported separately from truth-scored cause accuracy; they are not evidence of
counterfactual accuracy in a second physical world.
The credit calculation uses the existing abduction default, retaining recorded
prediction residuals during replay. It can therefore reproduce local baselines
while preserving effects that the world model did not learn. A valid replay alone
does not validate the causal model or justify removing a disturbance.

### Evaluation protocol

All runs use the existing frozen world-model checkpoint and arm MPPI settings in
`configs/tabletop.yaml`, with rendering disabled, a blue plate and green bowl,
and paired planner seeds. Attribute teaching uses 58000; clean surprise calibration
uses **58010–58017**. Development uses **59000** under all five conditions and all
three modes. Its interrupted run preserved eight completed trials and resumed the
remaining seven; the JSON retains the original provenance under `resumed_from`.
An earlier CPU calibration attempt was interrupted before producing episode data.
No cloud compute was used.

The frozen final evaluation uses layouts **66000–66003**, each under none, push,
heavier object, slippery object and camera shift. Each layout/condition is executed
separately as a direct goal, an English instruction and an Arabic instruction:
**60 physical trials, 20 matched layout/condition groups, four distinct layouts**.
They are paired comparisons, not 60 independent layouts. The placement instruction
is the held-out red-block/on/plate combination from the existing grammar curriculum.
Only one instruction goal is tested; task-general language robustness is unproven.
Disturbance RNG seeds are `layout_seed + 10000 + 100 * condition_index`, with the
condition order shown above. The seed-range scan found no pre-existing mentions
of these final layout or disturbance seeds in saved JSON/JSONL/Markdown results
and documentation. Parameters and thresholds are not tuned on final episodes.

Scoring compares cause identity to the disturbance event log. Abstentions count
as non-correct in unconditional cause accuracy; precision among issued reports
and reporting coverage are also shown. Weak-evidence abstention is a reporting
policy check, not successful identification of a hidden cause. In particular a
clean abstention does not earn a correct `none` label. Mass tolerance is **0.1 kg**,
camera-vector Euclidean tolerance **10 mm**, and onset tolerance **0.3 s**. The
scorer uses actual logged control-step times. Slip magnitude and push force have
no comparable causal-parameter ground truth in this benchmark and are not scored.
Parameter accuracy is conditional on an issued comparable estimate; denominators
are retained. Failure to infer an event is visible in the cause/coverage results.

### Reproduction and use

Use the project's Python from the repository root with `PYTHONPATH=.`. The
original commands are retained in each JSON. For a fresh reproduction without
overwriting those evidence files:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONIOENCODING='utf-8'
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes.py --seed 59000 --count 1 --split development --device cuda --output results/language_causes_development_repro.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes.py --seed 66000 --count 4 --split final --device cuda --teaching results/language_causes_development_repro.json --output results/language_causes_final_repro.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes_summary.py results/language_causes_final_repro.json --output results/language_causes_summary_repro.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe -m pytest (Get-ChildItem tests/test_language*.py).FullName tests/test_slice.py tests/test_cognition.py -q -p no:cacheprovider --basetemp=data/pytest_tmp
```

These reproduce the recorded seed set, not a new holdout. Choose unused seeds for
any future confirmatory evaluation and register all development reuse. Final
evaluation refuses to overwrite an existing result or resume across revisions;
interrupted development permits `--resume` with the same seeds and incomplete
output. The reports retain checkpoint/source hashes, commit, configuration,
package versions, fitted calibration, teacher provenance, learned templates,
per-trial beliefs, text, questions, answers, scores, and ground truth in separate
fields. Bitwise reproducibility across different GPU/software versions is not
claimed.

To use a saved language without importing the teacher:

```python
from pai.language.causal import ClauseLanguage, CausalExplainer, EpisodeEvidence

model = ClauseLanguage(report['learned_templates']['arabic'])
explainer = CausalExplainer(model, threshold=report['threshold'])
evidence = EpisodeEvidence.from_record(record, goal_observed=perceived_goal,
                                       credits=local_credit_records)
explanation = explainer.explain(evidence)
answer = explainer.answer(question_text, evidence)
```

`perceived_goal` must come from the agent's final observation; it must never be
filled with the simulator-scored `record.success`. `local_credit_records` contains
serialized `CreditResult` objects from the existing world-model replay.

### Frozen final results

Reports: `results/language_causes_development.json`,
`results/language_causes_final.json`, and `results/language_causes_summary.json`.
The final JSON contains all **60/60** completed physical trials, including all
three executions of the failed seed/condition. No final episode was replaced,
discarded, or used to tune the configuration. Each language column below uses its
own 20 instructed physical trials; translations of the other modes are not added
to its sample size.

| Metric | English | Arabic |
|---|---:|---:|
| Event-log cause correct, including abstentions in denominator | 15/20 (75%) | 15/20 (75%) |
| Cause reports issued | 17/20 (85%) | 17/20 (85%) |
| Correct among issued cause reports | 15/17 (88.2%) | 15/17 (88.2%) |
| Abstentions | 3/20 | 3/20 |
| Clean / disturbed abstentions | 2 / 1 | 2 / 1 |
| Perceived final goal report matches simulator scoring | 20/20 | 20/20 |
| Instructed task success | 19/20 (95%) | 19/20 (95%) |
| Paired direct-goal baseline success | 19/20 (95%) | 19/20 (95%) |
| Paired success disagreements with baseline | 0/20 | 0/20 |

| Condition | English cause correct / issued / trials | Arabic cause correct / issued / trials | Direct-goal success | English instruction success | Arabic instruction success |
|---|---:|---:|---:|---:|---:|
| None | 2 / 2 / 4 | 2 / 2 / 4 | 4/4 | 4/4 | 4/4 |
| Push | 4 / 4 / 4 | 4 / 4 / 4 | 4/4 | 4/4 | 4/4 |
| Heavier object | 4 / 4 / 4 | 4 / 4 / 4 | 4/4 | 4/4 | 4/4 |
| Slippery object | 1 / 3 / 4 | 1 / 3 / 4 | 3/4 | 3/4 | 3/4 |
| Camera shift | 4 / 4 / 4 | 4 / 4 / 4 | 4/4 | 4/4 | 4/4 |

The equal language scores reflect the same paired physical behavior and the same
semantic evidence policy. They establish no detected instruction penalty on this
small sample; they do not prove general equivalence, task-general competence, or
95% success over unseen layouts.

| Comparable parameter | Tolerance | English passes / estimates | Arabic passes / estimates | Mean absolute/vector error, each language |
|---|---:|---:|---:|---:|
| Extra mass | 0.1 kg | 4/4 | 4/4 | 0.001573 kg |
| Camera translation | 10 mm Euclidean | 4/4 | 4/4 | 0.3046 mm |
| Onset versus logged disturbance start | 0.3 s | 8/10 | 8/10 | 2.674 s |

The eight push/camera onset estimates all pass. The two issued slip-condition
onsets fail: errors are **13.38 s** and **13.08 s** (the latter also names the wrong
cause). The model infers a visible effect interval rather than the unobserved
initial friction change. No onset is invented for the remaining abstention or
`none` report. Large errors remain in the aggregate rather than being filtered.

| Question family | English | Arabic | What this score means |
|---|---:|---:|---|
| Why: correct cause against event log | 15/20 | 15/20 | Same cause evidence as episode explanation |
| Why did you drop: justified abstention | 20/20 | 20/20 | No supported attribution to a specific drop |
| Without the push: justified abstention | 20/20 | 20/20 | No disturbance-removal intervention exists |
| Open fingers early: model answers / abstentions | 17 / 3 | 17 / 3 | Local credit result availability |
| Open fingers early: valid-evidence policy followed | 20/20 | 20/20 | Model consistency, not physical counterfactual accuracy |

The audit script reconstructs **120/120 bilingual report/answer bundles** using
only each stored `belief` and learned templates, verifies there are exactly 60
distinct expected physical trials, and independently recomputes cause, goal, and
parameter scores from the separate raw truth fields. It confirms that the saved
words are supported by the saved agent evidence, including when that evidence is
wrong; it does not certify the cause inference itself. All recorded source hashes
matched the frozen final files after evaluation.

### Examples next to the scoring truth

These are verbatim saved outputs for the English-instruction executions; each
physical episode is rendered independently by both learned language models.
Ground truth in the first column is an evaluator annotation, never report input.

| Episode / scoring truth | English explanation | Arabic explanation |
|---|---|---|
| 66000, payload +0.447426 kg; goal reached | My evidence suggests that the object weighed more than expected. I estimated about 0.449 kg of extra mass. I updated my expected weight. My final observation indicates that the goal was reached. | تشير أدلتي إلى أن الجسم كان أثقل مما توقعت. قدرت الكتلة الإضافية بحوالي 0.449 كغ. حدثت توقعاتي للوزن. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |
| 66000, camera translation (-41.084, -2.843, -4.379) mm, logged at 1.78 s; goal reached | My evidence suggests that the camera moved. I estimated camera displacement (x, y, z) of about -41.5, -3.2, -4.2 mm. I estimated onset at planning step 17. I recalibrated my camera. My final observation indicates that the goal was reached. | تشير أدلتي إلى أن الكاميرا تحركت. قدرت إزاحة الكاميرا (س، ص، ع) بحوالي -41.5, -3.2, -4.2 مم. قدرت البداية عند خطوة التخطيط 17. أعدت معايرة الكاميرا. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |
| 66001, friction scale 0.05 from 0.02 s; goal failed | My evidence suggests that no unusual cause was detected. My final observation does not show the goal reached. | تشير أدلتي إلى أن لم أكتشف سببا غير معتاد. لا تظهر ملاحظتي الأخيرة تحقق الهدف. |
| 66003, friction scale 0.05; goal reached | I do not have enough evidence to identify the cause. I changed to a low carry with a central grasp. I lost my hold and returned to grasping 1 time(s). My final observation indicates that the goal was reached. | ليس لدي دليل كاف لتحديد السبب. غيرت الحمل ليكون منخفضا مع الإمساك من المركز. فقدت الإمساك وعدت إلى محاولة الإمساك 1 مرة. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |

### Failures and remaining limits

- Slip seed **66001** fails in all three execution modes at 149 planning steps.
  The final posterior assigns `none` probability 0.999998. Goal failure is reported
  correctly; the cause label is wrong against the friction event log.
- Slip seed **66002** succeeds but confidently selects `push` instead of slip.
  Its report also describes camera recalibration and low carrying because those
  actions actually occurred in the adaptation record. They are not evidence that
  either diagnosis was correct.
- Slip seed **66003** returns `unknown` and abstains on the cause. This is correct
  reporting-policy behavior but does not earn a correct cause-identification point.
  Two clean layouts also abstain because the monitor produced no posterior.
- Both slip onset estimates fail. Sink magnitude is a fitted residual quantity,
  not a simulator-labelled displacement with an independently scored tolerance.
- The confidence threshold does not fix the arm classifier's confident mistakes.
  Improving hidden-cause inference lies outside the permitted language files.
- Specific-drop attribution and removal-of-push counterfactuals remain unavailable;
  the interface exposes these gaps rather than fabricating answers. Local replay
  answers have no measured physical counterfactual accuracy.
- Cause phrases and question syntax are controlled supervised templates. The raw
  Arabic outputs retain awkward agreement/composition (including the `none`
  example); no human fluency rating or unrestricted bilingual competence is claimed.
- The sample has only four layouts and one placement instruction. Historical
  relation-Q&A/conversation scores, cross-situational word discovery, spoken audio,
  other manipulation goals, and noisy pixel perception remain unvalidated here.

The required regression command passed **71 tests** (language, slice, cognition).
Earlier attempts had six fixture setup errors from inaccessible Windows temporary
directories; using `--basetemp=data/pytest_tmp` resolved those infrastructure errors.
The new fast tests cover evidence allowlisting, independently taught surface forms,
unseen numeric/episode slots, missing cause vocabulary, weak/invalid posteriors,
nonfinite parameters, wrong-episode references, counterfactual validity/ambiguity,
false drop presuppositions, and reporting an actual adaptation despite an uncertain
final diagnosis. Existing tests and other stages were not edited.

## Review 1 revision: cross-situational associations (v2)

The v1 phrase-ID lookup above did **not** learn a cause-word grounding: observation
fields were unused and shared semantic key names supplied the answer. Its results
and limitations are preserved. V2 replaces that lookup with an observation-driven
association model and a larger, varied-goal evaluation. The old v1 final episodes
are explicitly **development/teaching data for v2**, never a v2 holdout.

### Anonymous belief grounding and numeric learning

`association.py` separates a given boundary adapter (`BeliefCoordinates`) from
the learner (`EvidenceLexicon`). The adapter shuffles six cause coordinates and
15 numeric feature coordinates with seed 72800. It retains the arm names only
for wiring at runtime and audit display. The learner receives arrays indexed by
integers, teacher phrase strings and teacher numeric observations. It receives
no arm variable names, simulator cause labels, source condition, or shared
`cause:...` / `estimate:...` identifiers. Runtime compatibility keys are created
**after** learning from the fitted associations, never used to fit them.

For a phrase, the learner accumulates expected posterior mass across paired
episodes. It ranks coordinates by conditional posterior mass times PMI relative
to the marginal coordinate frequency. A mapping needs at least 1.5 effective
posterior observations, positive PMI and a weighted-score margin above 0.1.
Competing phrases for the same coordinate abstain. A rare catch-all coordinate
cannot tie a repeatedly supported coordinate solely because both have high PMI.
Zero/missing, uninformative and contradictory observations do not supply a link.
Permutation tests reverse coordinates while keeping sentences fixed and verify
that the learned links reverse with the evidence. There is no string-equality
shortcut between a teacher cause name and an agent variable.

Teacher numbers are paired with all anonymous fitted numeric features. Matching
requires three distinct observed values, at least three valid pairs, correlation
above 0.5, normalized identity RMSE below 0.5, and a 0.05 error margin over the
next candidate. Teacher numeric spans, vector axes, scalar/vector shapes and the
assumption that the matching native field uses the teacher's numeric units are
**given**; field identity is **learned**. Distractor fields can use other units;
their units are not supplied as feature labels. We do not fit an arbitrary
scale that could turn a force feature into an extra-mass feature. Slip onset is
constant in the teaching truth and is not identifiable by this criterion, so v2
does not invent a numeric binding for it.

The teacher dictionaries, clause boundaries, number spans, report scaffolding,
action/goal narration, confidence policy and causal model remain given. English
and Arabic association and intent models are fitted separately from their own
sentences; neither loads the other's learned mapping. This learns associations
of supplied phrases, not unsupervised word segmentation or new grammar.

Teaching uses the 25 unique **baseline** episodes from the original development
and final JSONs: five conditions on layouts 59000 and 66000–66003. Paired language
duplicates are excluded. Files are read without modification. Classes are
interleaved for the learning curve. The complete learned mapping in both languages
is:

| Anonymous coordinate | Arm variable (audit only) | Learned English phrase | Independently learned Arabic phrase |
|---|---|---|---|
| 0 | push | an external force pushed my arm | قوة خارجية دفعت ذراعي |
| 1 | camera_shift | the camera moved | الكاميرا تحركت |
| 2 | heavier_object | the object weighed more than expected | الجسم كان أثقل مما توقعت |
| 3 | slippery_object | the object slipped relative to my fingers | الجسم انزلق بالنسبة إلى أصابعي |
| 4 | none | no unusual cause was detected | لم أكتشف سببا غير معتاد |
| 5 | unknown | no learned phrase | no learned phrase |

Learned numeric bindings (field names shown only to make the opaque indices
reviewable):

| Teacher phrase context | Anonymous numeric feature | Recovered agent field | Teaching pairs |
|---|---:|---|---:|
| Extra-mass sentence | 6 | heavier_object.extra_mass_kg | 5 |
| Camera displacement sentence | 2 | camera_shift.camera_offset_mm | 5 |
| Push onset sentence | 7 | push.onset | 5 |
| Camera onset sentence | 3 | camera_shift.onset | 5 |

The ordering-dependent teaching curve recovers 0/5, 4/5, 4/5, 4/5 and 5/5 cause
links after 5, 10, 15, 20 and 25 episodes respectively in each language. Four
numeric links become available at 15 episodes. This is an observed small-data
curve, not a guarantee that any 25 episodes suffice.

With exactly **five of 25 teacher cause labels wrong**, each of 20 seeds
72810–72829 draws an incorrect phrase uniformly from the other four labels.
Observations and numeric teacher measurements are unchanged; grouping them with
wrong cause phrases also stresses numeric binding. Both languages retain a mean
**4.15/5** correct cause links (range **3–5**) and **3.35/4** correct numeric links.
Randomly shuffling observations away from sentences (seeds 72900–72919) instead
recovers only **0.35/5** correct cause links on average and **zero** numeric links.
These are repeated, correlated training perturbations, not 100 independent
language trials. All mappings, supports, margins, correlations, noise replicates
and controls are saved in `results/language_causes_v2_checkpoint.json`.
As a further audit, refitting the complete 25-episode teaching set with a new
cause/feature permutation (seed 72899) again recovers 5/5 cause links and all four
numeric links in each language. This diagnostic is saved in
`results/language_causes_v2_permutation_audit.json`; it does not replace or tune
the frozen checkpoint used for final evaluation.

The initial development attempt is retained in
`results/language_causes_v2_development.json`. Its unweighted PMI tied slip with
the catch-all, and a strict support cutoff rejected 1.99998 effective clean
observations. The revised development diagnostics are retained in
`results/language_causes_v2_development_frozen.json`. Model choices were frozen
before the fresh v2 physical run.

### Varied goals, held-out wording and evaluation controls

The four evaluated instruction combinations are red block on blue plate, yellow
block inside green bowl, blue block on blue plate, and red block inside green
bowl. Every exact combination is removed from **all intents** in each language's
instruction-teaching curriculum. Tests verify that all four still parse exactly.
Ten layouts cycle through these four goals in counts 3, 3, 2 and 2, with the same
assignment under every condition and execution mode. Payload/friction
disturbances follow the instructed subject rather than always affecting red.
Pushes affect the hand and camera shifts affect the common view as before.

Each question intent has three teaching phrasings. Three further teaching
examples label ambiguous multi-intent questions as unclear. A separately fitted
TF-IDF nearest-example classifier requires cosine similarity at least 0.25 and a
0.08 margin between intents; unknown tokens and ambiguity can abstain. Initial
development wording diagnostics are explicitly not final tests. Final evaluation
uses **12 new, never-taught wordings per language**: two per supported question
family and four ambiguous/unrelated requests. Each unique wording is counted
once for the wording accuracy and Wilson interval; repeating it across episodes
does not enlarge that sample. Evidence-conditioned answers and intent errors are
also retained for every episode. A wrong intent is not silently counted correct
just because its answer happens to abstain.

The final layout seeds are **67000–67009**, giving **50 cases per mode, 150
separate physical executions**, with 10 layouts in each of five conditions.
Calibration uses 67400–67407 and attribute teaching 67420. Disturbance RNG seeds
are `78000 + 100 * condition_index + layout_index`. A pre-run search of earlier
saved results found only incidental matching digits in chess position hashes,
not prior use of these seeds. The v1 checkpoint and language results remain
unchanged. This seed set is used once for the frozen model; its errors do not
trigger further training or threshold changes.

The GPU is shared with walking training. Evaluation uses **64 MPPI samples**,
two planning iterations and one CPU thread, with sequential physical trials and
no parallel evaluation workers. The direct-goal baseline uses exactly the same
planning budget, goals, seeds and disturbances. This is a new benchmark protocol;
its results must not be interpreted as a direct improvement over v1's 256-sample,
single-goal result. World-model weights are unchanged.

Onset scoring now matches **only the event of the agent's inferred cause**. If
no unique matching event exists, the claim is marked unmatched and no timing
error is fabricated against another event. A multi-event regression test places
a push and friction change at different times and checks this behavior. Scoring
separately decodes the spoken phrase's meaning, so a wrong learned grounding
cannot receive credit just because the underlying latent field has the right
name. Numeric reports use only the learned feature binding and current beliefs.

All binary-rate tables below use **Wilson 95% intervals**. Overall episode-rate
intervals are descriptive binomial intervals: the same 10 layouts are reused
across conditions and the three modes are paired, so they do not establish 50
independent layouts or a statistically proven equivalence. Question intervals
use 12 unique authored wordings and do not imply open-language population
coverage. Noise-learning means above are not binomial rates.

### V2 commands and regression checks

Original evidence files are never overwritten. All commands run from the main
checkout using the project Python and `PYTHONPATH=.`; no Git state-changing
commands are part of this workflow. To reproduce with new output names:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONIOENCODING='utf-8'
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes_v2_train.py --output results/language_causes_v2_checkpoint_repro.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes_v2_eval.py --checkpoint results/language_causes_v2_checkpoint_repro.json --seed 67000 --count 10 --samples 64 --device cuda --output results/language_causes_v2_final_repro.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_causes_v2_summary.py --input results/language_causes_v2_final_repro.json --output results/language_causes_v2_summary_repro.json
$testTemp = 'data/pytest_language_v2_' + [guid]::NewGuid().ToString()
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/language_test_runner.py --basetemp $testTemp
```

The recorded evaluation used `results/language_causes_v2_checkpoint.json` and
`results/language_causes_v2_final.json`. Reusing 67000–67009 is a reproduction,
not another fresh holdout. New scientific comparisons need unused seeds.

The complete unchanged language/slice/cognition regression selection passes
**79 tests**, including coordinate/feature permutations, missing or conflicting
observations, compositional resolution of every held-out goal, unseen-wording
separation, question abstention, spoken-meaning grading, event-specific onset scoring and Wilson bounds
for all-zero/all-one outcomes. The first v2 permutation test exposed a shared-list
fixture error (four in-place reversals cancelled); giving each fixture its own
posterior list fixed that test data, without changing the learner.

On the managed Windows executor, ordinary pytest attempts reached 71 passed and
six fixture setup errors: Python's `mkdir(mode=0o700)` created temporary
directories inaccessible to the restricted execution token, including under
`data/pytest_tmp`. `language_test_runner.py` runs the same test files and only
lets **new** pytest temporary directories under its explicitly supplied fresh
workspace base inherit their normal parent permissions. It does not change any
existing ACL, test expectation, production permission or product behavior. The
successful final command used `data/pytest_language_v2_inherited_final_20261028`.

### V2 frozen final results

All **150/150** trials completed: 50 matched layout/condition cases in each of the direct-goal, English and Arabic modes. There were **123 successful and 27 failed physical executions**. No failures were replaced or used to tune the frozen model. Counts below show `k/n [Wilson 95% interval]`.

| Metric | English | Arabic |
|---|---:|---:|
| Instruction goal success | 41/50 [69.2–90.2%] | 41/50 [69.2–90.2%] |
| Matched direct-goal baseline | 41/50 [69.2–90.2%] | 41/50 [69.2–90.2%] |
| Cause correct against event log | 37/50 [60.4–84.1%] | 37/50 [60.4–84.1%] |
| Cause reporting coverage | 40/50 [67.0–88.8%] | 40/50 [67.0–88.8%] |
| Correct among issued cause reports | 37/40 [80.1–97.4%] | 37/40 [80.1–97.4%] |
| Perceived goal report matches truth | 50/50 [92.9–100.0%] | 50/50 [92.9–100.0%] |
| New question wording: intent correct | 9/12 [46.8–91.1%] | 9/12 [46.8–91.1%] |

Each language has **10 abstentions**, including eight clean cases without sufficient cause evidence and two disturbed cases. There are **zero paired task-success disagreements** with the direct-goal baseline. These are paired comparisons under one fixed planning budget, not proof that language has no cost on every task.

| Condition | Direct goal | English instruction | Arabic instruction | English cause correct | Arabic cause correct |
|---|---:|---:|---:|---:|---:|
| none | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 2/10 [5.7–51.0%] | 2/10 [5.7–51.0%] |
| push | 8/10 [49.0–94.3%] | 8/10 [49.0–94.3%] | 8/10 [49.0–94.3%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] |
| heavier_object | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] |
| slippery_object | 6/10 [31.3–83.2%] | 6/10 [31.3–83.2%] | 6/10 [31.3–83.2%] | 5/10 [23.7–76.3%] | 5/10 [23.7–76.3%] |
| camera_shift | 7/10 [39.7–89.2%] | 7/10 [39.7–89.2%] | 7/10 [39.7–89.2%] | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] |

| Held-out goal | Direct goal | English instruction | Arabic instruction |
|---|---:|---:|---:|
| `('in', 'red', 'bowl')` | 8/10 [49.0–94.3%] | 8/10 [49.0–94.3%] | 8/10 [49.0–94.3%] |
| `('in', 'yellow', 'bowl')` | 14/15 [70.2–98.8%] | 14/15 [70.2–98.8%] | 14/15 [70.2–98.8%] |
| `('on', 'blue', 'plate')` | 6/10 [31.3–83.2%] | 6/10 [31.3–83.2%] | 6/10 [31.3–83.2%] |
| `('on', 'red', 'plate')` | 13/15 [62.1–96.3%] | 13/15 [62.1–96.3%] | 13/15 [62.1–96.3%] |

The goal table pools conditions for the same assigned layouts; those denominators are not independent new layouts. Instruction resolution matched the intended goal in **50/50 cases per language**. Actual disturbance activation was checked from the event log: all 40 nominally disturbed layout/condition cases had a logged disturbance in each mode.

| Learned numeric meaning | English: correct / matched event | Arabic: correct / matched event | Unmatched claims, each language | Mean error, each language |
|---|---:|---:|---:|---:|
| Extra mass (0.1 kg tolerance) | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 0 | 0.003788 kg |
| Camera offset (10 mm tolerance) | 10/10 [72.2–100.0%] | 10/10 [72.2–100.0%] | 0 | 0.281237 mm |
| Onset (0.3 s tolerance) | 20/20 [83.9–100.0%] | 20/20 [83.9–100.0%] | 3 | 0.032000 s |

Onset was issued 23 times per language. Twenty claims had a matching push/camera event and passed; three push claims during slip had **no matching push event** and were left unmatched, without a fabricated error against the friction-change time. Those three wrong causes remain errors in the main cause score. No slip-onset binding was learned, so v2 makes no slip-onset claim. The conditional numeric results therefore do not establish accurate timing for every disturbance.

| Question intent / wording subset | English | Arabic |
|---|---:|---:|
| why | 2/2 [34.2–100.0%] | 2/2 [34.2–100.0%] |
| drop | 0/2 [0.0–65.8%] | 1/2 [9.5–90.5%] |
| cf:remove_push | 2/2 [34.2–100.0%] | 1/2 [9.5–90.5%] |
| cf:open_early | 2/2 [34.2–100.0%] | 2/2 [34.2–100.0%] |
| None | 3/4 [30.1–95.4%] | 3/4 [30.1–95.4%] |

On the same 12 new wordings, the unchanged v1 exact-match model scores **4/12 [13.8–60.9%]** in each language: it rejects all eight positive-intent wordings and correctly abstains on all four negatives. V2 scores **9/12 [46.8–91.1%]** in each language. This is a descriptive comparison on a small authored set; no population-wide language improvement or significance claim is made.

English misses both new drop wordings by abstaining. Arabic misreads one drop question as general why and abstains on one remove-push wording. Both classifiers label one mixed push/drop request as drop; its downstream drop-attribution answer still abstains, but its **intent classification is scored wrong**. The four negative wordings all receive abstaining replies on all episodes; this does not erase those intent errors. Specific-drop attribution and removing-push counterfactuals remain unsupported. Opening-early replies are local model predictions where a unique valid credit record exists, not measured alternate-world truth.

The final audit replayed **300/300** bilingual explanation/answer bundles from saved beliefs and checkpoint mappings, recomputed scores from separate raw event logs, verified complete nonduplicate trial keys, and confirmed no frozen runtime source drift. The v1 source data SHA-256 values still match the original files.

### V2 examples and retained failures

These are unedited saved outputs. The teacher/scorer truth column is never an input to report generation. Feature indices in numeric claims refer to the learned bindings above.

| Goal / evaluator truth | English | Arabic |
|---|---|---|
| Seed 67001, `in, yellow, bowl`; goal success `True`; `{"t":1,"type":"disturbance_start","disturbance":"payload","params":{"body":"yellow","mass":0.4730678025279621}}` | My evidence suggests that the object weighed more than expected. I updated my expected weight. My final observation indicates that the goal was reached. I estimated about 0.478 kg of extra mass. | تشير أدلتي إلى أن الجسم كان أثقل مما توقعت. حدثت توقعاتي للوزن. تشير ملاحظتي الأخيرة إلى تحقق الهدف. قدرت الكتلة الإضافية بحوالي 0.478 كغ. |
| Seed 67006, `on, blue, plate`; goal success `False`; `{"t":168,"type":"disturbance_start","disturbance":"camera_shift","params":{"offset":[-0.020337632325861478,-0.050976664188062806,0.003203635987926494]}}` | My evidence suggests that the camera moved. I recalibrated my camera. My final observation does not show the goal reached. I estimated camera displacement (x, y, z) of about -20.3, -50.9, 3.3 mm. I estimated onset at planning step 33. | تشير أدلتي إلى أن الكاميرا تحركت. أعدت معايرة الكاميرا. لا تظهر ملاحظتي الأخيرة تحقق الهدف. قدرت إزاحة الكاميرا (س، ص، ع) بحوالي -20.3, -50.9, 3.3 مم. قدرت البداية عند خطوة التخطيط 33. |
| Seed 67005, `in, yellow, bowl`; goal success `True`; `{"t":1,"type":"disturbance_start","disturbance":"friction","params":{"body":"yellow","scale":0.05}}` | My evidence suggests that an external force pushed my arm. I changed to a low carry with a central grasp. I lost my hold and returned to grasping 3 time(s). My final observation indicates that the goal was reached. I estimated onset at planning step 99. | تشير أدلتي إلى أن قوة خارجية دفعت ذراعي. غيرت الحمل ليكون منخفضا مع الإمساك من المركز. فقدت الإمساك وعدت إلى محاولة الإمساك 3 مرة. تشير ملاحظتي الأخيرة إلى تحقق الهدف. قدرت البداية عند خطوة التخطيط 99. |
| Seed 67002, `on, blue, plate`; goal success `True`; `{"t":1,"type":"disturbance_start","disturbance":"friction","params":{"body":"blue","scale":0.05}}` | I do not have enough evidence to identify the cause. I changed to a low carry with a central grasp. I lost my hold and returned to grasping 1 time(s). My final observation indicates that the goal was reached. | ليس لدي دليل كاف لتحديد السبب. غيرت الحمل ليكون منخفضا مع الإمساك من المركز. فقدت الإمساك وعدت إلى محاولة الإمساك 1 مرة. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |

Retained physical failures, identical across the three modes:

- `push`: 67006, 67008.
- `slippery_object`: 67006, 67007, 67008, 67009.
- `camera_shift`: 67002, 67006, 67007.

All three slip/push confusions (67003, 67005, 67009) are confidently wrong arm inferences expressed by correctly learned words. Slip cases 67002 and 67007 abstain. Correct cause reports do not guarantee successful control: camera-shift cases 67002, 67006 and 67007 report the shift but fail the task. The reporting layer does not claim that an identified episode disturbance necessarily caused every individual failure or drop.

The review requests are implemented, with these remaining limits: five-layout teaching data; only four instructed goals; a shallow intent model with 3/12 wording errors per language; awkward inherited controlled Arabic phrasing; approximate, uncalibrated arm cause posteriors; no learned slip-onset quantity; no verified physical counterfactual accuracy; and no speech or pixel-perception validation. Better word grounding has not repaired the arm controller or cause model. Changes to those other stages were outside this task.

Final artifacts: `results/language_causes_v2_checkpoint.json`, `results/language_causes_v2_final.json`, and `results/language_causes_v2_summary.json`, alongside both retained v2 development attempts and all unchanged v1 results.

## V3: silence of the surprise monitor as learned evidence

In v2, 8 of 10 clean episodes abstained: when the arm's surprise monitor never fires, the
agent has no cause posterior, so the reporter said "I do not have enough evidence". But
silence is evidence. `SilenceModel` (`pai/language/association.py`) learns what it means
from the teaching episodes only: how often the monitor stays silent with no disturbance
(11/15) and with any disturbance (0/60), with a Jeffreys prior, turned into
P(no disturbance | silent) = **0.957** by Bayes' rule. When the posterior is empty, the
explainer uses that probability for `none` and spreads the rest over the other causes;
the existing 0.9 reporting threshold and the taught phrase for `none` then apply as
before. Each explanation records `evidence_source` (`cause_monitor` or `silent_monitor`).

What is given: the pooling of all disturbances into one "disturbed" class, the prior and
the threshold. What is learned: the two silence rates, from teacher labels on teaching
episodes. The teaching set for v3 is the v2 development episodes **plus the v2 final
episodes (seeds 67000-67009)**, so those seeds are no longer a holdout; v3 is evaluated
on new seeds **68000-68009** with disturbance draws from base 79000 (v2 used 78000).

```powershell
$env:PYTHONPATH='.;scripts'
$env:PYTHONIOENCODING='utf-8'
python scripts/language_causes_v2_train.py --output results/language_causes_v3_checkpoint.json   # sources as in the checkpoint
python scripts/language_causes_v2_eval.py --checkpoint results/language_causes_v3_checkpoint.json --seed 68000 --count 10 --disturbance-base 79000 --output results/language_causes_v3_final.json
python scripts/language_causes_v2_summary.py --input results/language_causes_v3_final.json --output results/language_causes_v3_summary.json
python scripts/language_causes_v3_paired.py --output results/language_causes_v3_paired.json
```

### V3 frozen final results

All 150/150 trials completed (50 layout/condition cases × direct goal, English, Arabic).
The summary replayed all **300/300** bilingual bundles from the saved beliefs and
recomputed every score from the separate event logs. English and Arabic give identical
counts. `k/n [Wilson 95%]`:

| Metric | English and Arabic (each) |
|---|---:|
| Instruction goal success | 43/50 [73.8-93.0%] |
| Matched direct-goal baseline | 43/50 [73.8-93.0%] (0 paired disagreements) |
| Cause correct against event log | **41/50 [69.2-90.2%]** |
| Cause reporting coverage | 50/50 [92.9-100%] (no abstentions) |
| Perceived goal report matches truth | 50/50 [92.9-100%] |
| End-to-end "why" answers correct | 82/100 [73.3-88.3%] |
| New question wording: intent correct | 9/12 [46.8-91.1%] (same model as v2) |

| Condition | Cause correct (each language) | Of which via silent monitor |
|---|---:|---:|
| none | **9/10** [59.6-98.2%] | 8 correct of 8 silent |
| push | 10/10 | 0 |
| heavier_object | 7/10 | 1 silent, wrongly `none` |
| slippery_object | 5/10 | 1 silent, wrongly `none` |
| camera_shift | 10/10 | 0 |

**Paired comparison** (`results/language_causes_v3_paired.json`): the frozen v2 and v3
explainers replayed on the *same* v3 episodes. v2 gets 33/50 (10 abstentions, clean
cases 1/10); v3 gets 41/50 (0 abstentions). v3 is right where v2 was wrong in 8 episodes,
all clean, and never the reverse (sign test p = 0.008, each language). The disturbed
conditions are unchanged, since silence is only used when the monitor never fired.

The price of no abstentions: precision among issued reports falls from v2's 37/40
(different seeds) to 41/50. On these seeds the monitor was silent in 2/40 disturbed
cases (68009 heavier object, 68005 slippery object), and the agent then said "no unusual
cause was detected" for a real disturbance; teaching had 0/60 such cases, so the learned
0.957 is optimistic on 15 + 60 teaching episodes. The other wrong reports come from the
arm cause monitor itself, not from the language layer: clean 68001 reported a push;
heavier 68002/68003 and slippery 68003/68006/68009 reported a push; slippery 68004 gave
a confident `none` posterior. Numeric claims: mass 7/7, camera offset 10/10, onset 20/20
with matching events (6 onset claims had no matching push/camera event and stay
unmatched; their causes are counted wrong above).

Examples (unedited saved outputs):

| Seed / truth | English | Arabic |
|---|---|---|
| 68000, clean, success | My evidence suggests that no unusual cause was detected. My final observation indicates that the goal was reached. | تشير أدلتي إلى أن لم أكتشف سببا غير معتاد. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |
| 68009, heavier object, failed (silent monitor) | My evidence suggests that no unusual cause was detected. My final observation does not show the goal reached. | تشير أدلتي إلى أن لم أكتشف سببا غير معتاد. لا تظهر ملاحظتي الأخيرة تحقق الهدف. |

The Arabic `none` sentence is ungrammatical: the taught phrase "لم أكتشف" is placed in the
controlled frame "تشير أدلتي إلى أن", which needs a pronoun before a verb clause ("أنني لم
أكتشف"). This is a limit of the template frame, not of the learned cause meaning. Tests: the language/slice/cognition
selection passes **80 tests** (one new `SilenceModel` test).

Artifacts: `results/language_causes_v3_checkpoint.json`, `results/language_causes_v3_final.json`,
`results/language_causes_v3_summary.json`, `results/language_causes_v3_paired.json`.

## V4: learned Arabic composition and calibrated absence claims

V4 retains the frozen v3 anonymous-coordinate cause and quantity associations, question classifier, arm controller, and surprise monitor. It adds independently taught English/Arabic surface composition and new calibration of monitor silence and explicit `none` decisions. Event logs are supplied only to the teacher and scorer. V4 scoring decodes the actually rendered `surface_phrase`, including the hedge, rather than trusting a latent cause key or an old phrase spelling. Runtime explanations take `EpisodeEvidence` and frozen learned parameters, never truth or disturbance parameters. Original v1–v3 artifacts and documentation are unchanged.

### Design: learned versus given

Given: the teacher’s grammatical complete sentences and corrected phrases, semantic evidence schema, balanced five-condition deployment prior, independent Jeffreys Beta(0.5,0.5) likelihood priors, minimum claim probability 0.5 and confident threshold 0.95. Arabic morphology, vocabulary corrections, and the count-label construction are teacher supervision, not independently discovered grammar. The teacher supplies both noun complements and first/third-person verb complements. The learner removes the paired phrase from each sentence, induces distinct frames, and learns each phrase’s frame index without receiving a clause-type label, cause name, or hand-written complementizer-selection rule. Untaught compositions are rejected rather than guessed. This is controlled phrase/frame learning; it does not generalize inflection to arbitrary Arabic. The existing cause-coordinate links remain those learned cross-situationally in v3, independently per language.

Learned anew: phrase-to-frame choices (including `أنني` for both first-person `none` and hedge clauses, `أنّ` before noun complements, and teacher examples with `أنه`/`أنها`), a probability sentence template induced from two numeric teacher examples, and per-condition frequencies of silence and explicit `none`. The surface phrase corrections and hedge wording are given by the teacher. Numeric probability slots come from the learned calibration model. English disturbed reports remain byte-identical to v3; English absence claims gain matched hedging and a numeric probability. Arabic count reporting uses a grammatical count label for every cardinal rather than an incorrectly invariant `مرة` after arbitrary numerals.

For condition c and event e, the learned posterior-predictive likelihood is `(count(e,c)+0.5)/(n(c)+1)`. Bayes’ rule normalizes these five likelihoods with an explicitly balanced prior. Pooling four unseen disturbance types into one near-zero pooled likelihood is avoided. Silence and an explicit monitor `none` are calibrated separately. Credible intervals propagate Beta likelihood uncertainty (20,000 fixed diagnostic draws, seed 70420); reported probabilities are ratios of posterior-predictive likelihoods, not lower confidence bounds. Claims below 0.5 abstain, 0.5–0.95 hedge, and at least 0.95 use the confident phrase. Neither thresholds nor priors were selected using final episodes. The probability depends on the balanced task mixture and is not a deployment-invariant safety guarantee.

### Teaching, freeze, seeds, and reproducibility

The teaching collection used the unchanged v3 explainer while v4 files were being developed; its saved v3 outputs and question answers were all replayed before fitting, so development-file hash changes are not claimed to be a frozen v4 run. New teaching/calibration: layouts **70000–70019**, disturbance base **81000**, 20 episodes per condition / 100 physical trials, four goals balanced across layouts. Existing monitor residual calibration remains seeds 67400–67407 and attribute teaching seed 67420, matching v3; these are development-only. An initial frozen attempt on 69000–69019 / base 80000 was stopped and invalidated when printed inventory review found question-mark corruption from Windows shell encoding in several new Arabic strings. Its checkpoint, partial workers, logs, tests, and inventory are retained under `language_causes_v4_invalidated_69000_*`; they are excluded from all final scores. Statistical parameters were not tuned after this discovery. Explicit UTF-8 source writing and an independent ASCII-escaped Unicode regression test repair and guard the text. V4 was then frozen again before any replacement final simulation. Fresh final: **69020–69039**, disturbance base **82000**, 100 physical episodes collected by two disjoint modest GPU workers (64 planning samples each); worker files are merged only after complete nonduplicate key checks and matching calibration/source hashes. Seed ranges were checked against prior language result seed fields. Each saved direct-goal episode produces both languages; v3 and v4 replay the identical evidence. These are 100 shared physical trials, not 200 independent samples. Instruction parsing, control, and language-instruction execution were not changed or separately re-evaluated here.

Run from the repository root using the project Python:

```powershell
$env:PYTHONPATH='.;scripts'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe scripts/language_causes_v4_eval.py --teaching --checkpoint results/language_causes_v3_checkpoint.json --seed 70000 --count 20 --disturbance-base 81000 --output results/language_causes_v4_teaching.json
.venv/Scripts/python.exe scripts/language_causes_v4_train.py
.venv/Scripts/python.exe scripts/language_causes_v4_final.py
.venv/Scripts/python.exe scripts/language_causes_v4_report.py --docs
.venv/Scripts/python.exe scripts/language_causes_v4_tests.py --basetemp data/pytest_language_v4_utf8_inherited_20261028 --output results/language_causes_v4_tests_frozen.json
```

Use a new output name to collect new experiments; `--resume` continues only incomplete matching frozen runs. Do not rerun final seeds for model selection. The report also checks source/model hashes and exactly replays all 200 bilingual explanations and all saved question answers. Tests include exhaustive controlled Arabic clause inventory matching over all causes, goal states, action frames, and representative cardinalities, plus every `none`/hedge complementizer. On this Windows installation pytest’s mode-0700 temporary directories are inaccessible; the test wrapper changes only new project test-directory creation to inherited permissions, restores `os.mkdir` afterward, and never deletes an existing directory.

### Learned absence likelihoods

| Condition | Teaching n | Silent count | P(silent given condition) | Explicit-none count | P(explicit none given condition) |
|---|---:|---:|---:|---:|---:|
| none | 20 | 10 | 0.5000 | 8 | 0.4048 |
| push | 20 | 0 | 0.0238 | 0 | 0.0238 |
| heavier_object | 20 | 0 | 0.0238 | 0 | 0.0238 |
| slippery_object | 20 | 0 | 0.0238 | 1 | 0.0714 |
| camera_shift | 20 | 0 | 0.0238 | 0 | 0.0238 |

`silent`: P(no disturbance given evidence) = **0.8400**, posterior credible 95% [0.6348, 0.9770].

`explicit_none`: P(no disturbance given evidence) = **0.7391**, posterior credible 95% [0.5000, 0.9314].

### Frozen final paired results

Each language has the same counts below because the evidence and semantic mappings agree. Wilson 95% intervals describe these sampled cases. Hedged reports count as issued `none` hypotheses and can still be wrong; the hedge never erases an error.

| Metric, each language | V3 | V4 |
|---|---:|---:|
| Correct cause / all episodes | 83/100 [74.5–89.1%] | 83/100 [74.5–89.1%] |
| Issued cause reports / all episodes | 95/100 [88.8–97.8%] | 95/100 [88.8–97.8%] |
| Correct / issued reports | 83/95 [79.2–92.6%] | 83/95 [79.2–92.6%] |
| Correct / confident none reports | 17/18 [74.2–99.0%] | 0/0 (not estimable) |
| Direct-goal task success | 81/100 [72.2–87.5%] | 81/100 [72.2–87.5%] |
| Hedged none reports | 0 | 18 |

Paired sign test, each language: only v4 correct **0**, only v3 correct **0**, two-sided p = **1**. These are identical paired outcomes in two languages, not independent replications.

| Condition | V3 cause correct | V4 cause correct |
|---|---:|---:|
| none | 17/20 [64.0–94.8%] | 17/20 [64.0–94.8%] |
| push | 20/20 [83.9–100.0%] | 20/20 [83.9–100.0%] |
| heavier_object | 17/20 [64.0–94.8%] | 17/20 [64.0–94.8%] |
| slippery_object | 9/20 [25.8–65.8%] | 9/20 [25.8–65.8%] |
| camera_shift | 20/20 [83.9–100.0%] | 20/20 [83.9–100.0%] |

### Calibration and reliability

Primary Brier score is the binary forecast P(no disturbance) against the event-log absence of a disturbance, over every episode (lower is better). It is not a multiclass Brier score. The silent subset measures the targeted absence forecasts. A second top-label reliability analysis compares each reported confidence with correctness of the issued cause decision; abstentions are excluded from that conditional analysis and retained in the coverage score. Confident `none` precision uses a shared at-least-0.95 cutoff (and v4’s confident wording). A zero denominator is reported as not estimable, not perfect precision.

| Calibration metric, each language | V3 | V4 |
|---|---:|---:|
| Binary none, all episodes: Brier (n) | 0.040189 (100) | 0.041513 (100) |
| Binary none, silent subset: Brier (n) | 0.001816 (13) | 0.025600 (13) |
| Top-label report correctness: Brier (n) | 0.126515 (95) | 0.127909 (95) |

| Forecast | Version | Probability bin | n | Mean stated probability | Observed frequency / accuracy [Wilson 95%] |
|---|---|---|---:|---:|---|
| No-disturbance forecast | v3 | [0.0, 0.2] | 82 | 0.0000 | 3/82 [1.3–10.2%] |
| No-disturbance forecast | v3 | [0.2, 0.4] | 0 | — | 0/0 (not estimable) |
| No-disturbance forecast | v3 | [0.4, 0.6] | 0 | — | 0/0 (not estimable) |
| No-disturbance forecast | v3 | [0.6, 0.8] | 0 | — | 0/0 (not estimable) |
| No-disturbance forecast | v3 | [0.8, 1.0] | 18 | 0.9690 | 17/18 [74.2–99.0%] |
| No-disturbance forecast | v4 | [0.0, 0.2] | 82 | 0.0000 | 3/82 [1.3–10.2%] |
| No-disturbance forecast | v4 | [0.2, 0.4] | 0 | — | 0/0 (not estimable) |
| No-disturbance forecast | v4 | [0.4, 0.6] | 0 | — | 0/0 (not estimable) |
| No-disturbance forecast | v4 | [0.6, 0.8] | 5 | 0.7391 | 4/5 [37.6–96.4%] |
| No-disturbance forecast | v4 | [0.8, 1.0] | 13 | 0.8400 | 13/13 [77.2–100.0%] |
| Top-label report | v3 | [0.0, 0.2] | 0 | — | 0/0 (not estimable) |
| Top-label report | v3 | [0.2, 0.4] | 0 | — | 0/0 (not estimable) |
| Top-label report | v3 | [0.4, 0.6] | 0 | — | 0/0 (not estimable) |
| Top-label report | v3 | [0.6, 0.8] | 0 | — | 0/0 (not estimable) |
| Top-label report | v3 | [0.8, 1.0] | 95 | 0.9941 | 83/95 [79.2–92.6%] |
| Top-label report | v4 | [0.0, 0.2] | 0 | — | 0/0 (not estimable) |
| Top-label report | v4 | [0.2, 0.4] | 0 | — | 0/0 (not estimable) |
| Top-label report | v4 | [0.4, 0.6] | 0 | — | 0/0 (not estimable) |
| Top-label report | v4 | [0.6, 0.8] | 5 | 0.7391 | 4/5 [37.6–96.4%] |
| Top-label report | v4 | [0.8, 1.0] | 90 | 0.9769 | 79/90 [79.4–93.0%] |

### Complete controlled Arabic sentence inventory

The following enumerates all reachable sentence templates, plus the additional agreement examples taught to the composer (marked teaching-only). A paragraph is a composition of these sentences. `{value}` ranges over agent estimates, a nonnegative count label, or a numeric probability; infinitely many numeric spellings are represented by the template, not listed separately. No general-purpose Arabic generator or unrestricted paraphrasing is claimed. Definite noun complements retain the article; the indefinite subject after `أنّ` is taught as `قوةً خارجيةً`, and indefinite accusatives include `سببًا`, `شيئًا`, `اضطرابًا صغيرًا`, `ممكنًا`, `اعتمادًا`. Unvowelled standard spellings elsewhere do not assert fully vocalized case annotation.

| Sentence / template | Review note |
|---|---|
| تشير أدلتي إلى أنني لم أكتشف سببًا غير معتاد. | learned first-person verb frame; accusative spelling corrected |
| تشير أدلتي إلى أنني لم ألاحظ شيئًا غير معتاد، لكنّ اضطرابًا صغيرًا قد يمر دون أن ألاحظه. | learned first-person verb frame; accusative spelling corrected |
| تشير أدلتي إلى أنّ قوةً خارجيةً دفعت ذراعي. | given gender/number agreement; learned noun/pronoun complement frame |
| تشير أدلتي إلى أنّ الجسم كان أثقل مما توقعت. | given gender/number agreement; learned noun/pronoun complement frame |
| تشير أدلتي إلى أنّ الجسم انزلق بالنسبة إلى أصابعي. | given gender/number agreement; learned noun/pronoun complement frame |
| تشير أدلتي إلى أنّ الكاميرا تحركت. | given gender/number agreement; learned noun/pronoun complement frame |
| تشير أدلتي إلى أنها تحركت. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| تشير أدلتي إلى أنه تحرك. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| تشير أدلتي إلى أنّ الكتلة الإضافية جعلت الجسم أثقل. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| تشير أدلتي إلى أنّ الهدف تحقق. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| تشير أدلتي إلى أنّ المهمة تحققت. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| تشير أدلتي إلى أنّ الأجسام الأثقل تحركت. | given gender/number agreement; learned noun/pronoun complement frame (teaching-only) |
| ليس لدي دليل كافٍ لتحديد السبب. | corrected teacher wording/count construction |
| أعدت معايرة الكاميرا. | unchanged grammatical sentence |
| حدّثت توقعاتي بشأن الوزن. | corrected teacher wording/count construction |
| غيّرت أسلوب الحمل إلى حمل الجسم على ارتفاع منخفض مع الإمساك به من مركزه. | corrected teacher wording/count construction |
| خفضت سرعة الحمل إلى النصف. | unchanged grammatical sentence |
| فقدت الإمساك بالجسم. عدد مرات إعادة محاولة الإمساك: {value}. | corrected teacher wording/count construction |
| تشير ملاحظتي الأخيرة إلى تحقق الهدف. | unchanged grammatical sentence |
| لا تظهر ملاحظتي الأخيرة تحقق الهدف. | unchanged grammatical sentence |
| ليس لدي رصد نهائي للهدف. | unchanged grammatical sentence |
| لا أستطيع الإجابة عن هذا السؤال الافتراضي بالأدلة المتاحة من نموذجي. | corrected teacher wording/count construction |
| لا أستطيع تحديد سبب سقوط الجسم اعتمادًا على تقدير سبب الاضطراب في الحلقة بأكملها. | corrected teacher wording/count construction |
| في إعادة المحاكاة المحلية بنموذجي، منع هذا التغيير تحقق الهدف الفرعي المختبَر. هذا تنبؤ وليس نتيجة مرصودة. | corrected teacher wording/count construction |
| في إعادة المحاكاة المحلية بنموذجي، ظل تحقق الهدف الفرعي المختبَر ممكنًا مع هذا التغيير. هذا تنبؤ وليس نتيجة مرصودة. | corrected teacher wording/count construction |
| قدرت الكتلة الإضافية بحوالي {value} كغ. | unchanged; {value} is an agent estimate |
| قدرت إزاحة الكاميرا (س، ص، ع) بحوالي {value} مم. | unchanged; {value} is an agent estimate |
| قدرت البداية عند خطوة التخطيط {value}. | unchanged; {value} is an agent estimate |
| أقدّر احتمال عدم وجود اضطراب بنحو {value}٪. | new taught probability sentence; {value} is calibrated probability |

### Unedited examples next to scorer-only truth

| Seed / condition / event log / success | English | Arabic |
|---|---|---|
| 69020 / none / `[]` / True | My evidence suggests that I noticed nothing unusual, though a small disturbance could go unnoticed. I estimate the probability of no disturbance at about 84.0%. My final observation indicates that the goal was reached. | تشير أدلتي إلى أنني لم ألاحظ شيئًا غير معتاد، لكنّ اضطرابًا صغيرًا قد يمر دون أن ألاحظه. أقدّر احتمال عدم وجود اضطراب بنحو 84.0٪. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |
| 69020 / push / `[{"t":212,"type":"disturbance_start","disturbance":"push","params":{"body":"hand","force":[-1.0530107620729736,46.90457102026092,0.0]}}]` / True | My evidence suggests that an external force pushed my arm. My final observation indicates that the goal was reached. I estimated onset at planning step 42. | تشير أدلتي إلى أنّ قوةً خارجيةً دفعت ذراعي. تشير ملاحظتي الأخيرة إلى تحقق الهدف. قدرت البداية عند خطوة التخطيط 42. |
| 69020 / heavier_object / `[{"t":1,"type":"disturbance_start","disturbance":"payload","params":{"body":"red","mass":0.5478086754157062}}]` / True | I do not have enough evidence to identify the cause. I updated my expected weight. I lost my hold and returned to grasping 1 time(s). My final observation indicates that the goal was reached. | ليس لدي دليل كافٍ لتحديد السبب. حدّثت توقعاتي بشأن الوزن. فقدت الإمساك بالجسم. عدد مرات إعادة محاولة الإمساك: 1. تشير ملاحظتي الأخيرة إلى تحقق الهدف. |
| 69020 / slippery_object / `[{"t":1,"type":"disturbance_start","disturbance":"friction","params":{"body":"red","scale":0.05}}]` / False | My evidence suggests that the object slipped relative to my fingers. I lost my hold and returned to grasping 1 time(s). My final observation does not show the goal reached. | تشير أدلتي إلى أنّ الجسم انزلق بالنسبة إلى أصابعي. فقدت الإمساك بالجسم. عدد مرات إعادة محاولة الإمساك: 1. لا تظهر ملاحظتي الأخيرة تحقق الهدف. |
| 69020 / camera_shift / `[{"t":185,"type":"disturbance_start","disturbance":"camera_shift","params":{"offset":[0.020753049383231977,-0.029261979767575993,0.00979502688835004]}}]` / True | My evidence suggests that the camera moved. I recalibrated my camera. My final observation indicates that the goal was reached. I estimated camera displacement (x, y, z) of about 21.1, -29.3, 10.2 mm. I estimated onset at planning step 37. | تشير أدلتي إلى أنّ الكاميرا تحركت. أعدت معايرة الكاميرا. تشير ملاحظتي الأخيرة إلى تحقق الهدف. قدرت إزاحة الكاميرا (س، ص، ع) بحوالي 21.1, -29.3, 10.2 مم. قدرت البداية عند خطوة التخطيط 37. |

### Failures and limits

Calibration is assessed once on the new frozen set and is not guaranteed to improve every metric. The inherited non-`none` arm posteriors remain uncalibrated; false pushes and heavier/slippery confusions are outside this language-only change. A hedged “nothing unusual” is still scored wrong when a real disturbance occurred. The given balanced prior may be wrong in deployment, and the per-condition sample sizes leave broad uncertainty. Absence probabilities are conditional on the five taught conditions; unseen disturbance types and sensor failures are not calibrated here. No additional practice, controller change, monitor tuning, novel question-intent training, speech, or unrestricted Arabic morphology was performed.

- V4 cause errors/abstentions, `none`: [69030, 69032, 69035].
- V4 cause errors/abstentions, `push`: none.
- V4 cause errors/abstentions, `heavier_object`: [69020, 69025, 69027].
- V4 cause errors/abstentions, `slippery_object`: [69023, 69024, 69027, 69028, 69030, 69031, 69032, 69034, 69035, 69036, 69038].
- V4 cause errors/abstentions, `camera_shift`: none.

Physical task failures (shared by the reporting comparisons): `[('push', 69027), ('push', 69037), ('slippery_object', 69020), ('slippery_object', 69023), ('slippery_object', 69025), ('slippery_object', 69026), ('slippery_object', 69028), ('slippery_object', 69029), ('slippery_object', 69030), ('slippery_object', 69032), ('slippery_object', 69033), ('slippery_object', 69034), ('slippery_object', 69035), ('slippery_object', 69037), ('slippery_object', 69038), ('slippery_object', 69039), ('camera_shift', 69032), ('camera_shift', 69035), ('camera_shift', 69036)]`.

Artifacts: `results/language_causes_v4_teaching.json`, `_checkpoint.json`, `_final.json`, `_summary.json`, and `_tests_frozen.json`. Final summary includes all per-episode v3/v4 explanations, raw beliefs, separate truth, numeric forecasts, and reliability bins. The v4 evaluator records project-relative commands and frozen hashes; result files contain no absolute local paths.

### V4 final audit and interpretation

The complete gate passed **94 tests**. The final audit independently replayed **200 bilingual explanations and 2,400 question answers**, matched all **1,300 Arabic output texts** against the full reviewed inventory, and verified the verb-clause complementizer in every emitted Arabic absence report. All **82 English non-absence texts** were identical to v3. Frozen runtime, world-model, and original language-artifact hashes remain unchanged; all new result files pass the local-path privacy check. See `results/language_causes_v4_completion_audit.json` and `_encoding_repair_audit.json`.

**Negative calibration result:** binary Brier worsened from **0.040189 to 0.041513**, and silent-subset Brier worsened from **0.001816 to 0.025600**. All 13 silent final episodes happened to be clean, making v4's conservative 0.84 less accurate on that subset than v3's 0.957. The five explicit-none cases were four clean and one disturbed; v4 stated 0.739 for each. Overall cause decisions were unchanged (83/100, coverage 95/100, precision 83/95, sign-test p=1). V4 hedged all 18 absence reports, including the one disturbed case; that case remains a scored error. It issued **zero confident absence claims**, so their precision is **not estimable**, not 100%. These results support the Arabic repair and a more cautious reporting policy; they do **not** demonstrate improved held-out probability calibration. No parameters were retuned after this negative result.

Reliability uses unrounded JSON confidence/forecast values; absence sentences display one decimal percent. Wilson intervals are descriptive episode-level intervals: 20 layouts are reused across five conditions, so no layout-cluster correction is claimed. The 48 saved episodes from the encoding-invalidated attempt remain excluded, along with its incomplete in-flight episodes. No other stages or default language entry points were changed, and no git-state-changing commands were run.
