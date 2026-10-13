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
