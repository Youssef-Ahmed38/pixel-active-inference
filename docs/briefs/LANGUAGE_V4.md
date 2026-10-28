# Brief: Stage 6 language v4 — correct Arabic and a calibrated "nothing unusual"

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (active inference, hidden-cause
inference, learned rather than scripted, honest evaluation). This task improves the agent's causal
explanations in English and Arabic (Stage 6, v3 merged in PR #2).

## 1. Read first

| File | What it is |
|---|---|
| `docs/LANGUAGE_STAGE6.md`, sections "V2" and "V3: silence of the surprise monitor as learned evidence" | the current explainer, its results on seeds 68000-68009 and its known problems |
| `pai/language/association.py` (`SilenceModel`), `pai/language/*` | the explainer, the learned cause phrases, the sentence frames |
| `scripts/language_causes_v2_train.py`, `_v2_eval.py`, `_v2_summary.py`, `_v3_paired.py` | training, evaluation, summary, paired comparison |
| `results/language_causes_v3_*.json` | v3 checkpoint and frozen results |

## 2. The two problems

1. **Arabic grammar.** The `none` sentence is ungrammatical: "تشير أدلتي إلى أن لم أكتشف سببا غير
   معتاد" puts a verb clause after "أن" (it needs "أنني لم أكتشف ..."). The doc also notes other awkward
   agreement and composition in the controlled Arabic frames. Fix the composition properly, not by
   patching one string: the frame must choose the right complementizer form (أن + noun / أنّ + noun
   with accusative agreement / أنني، أنه، أنها + verb clause), gender and number agreement (e.g. الكتلة
   الأثقل vs الجسم الأثقل، الهدف تحقق vs المهمة تحققت), definiteness, and the case ending on
   indefinite accusatives (سببًا). Prefer learning the frame choice from the teacher's sentences
   (teacher sentences with both clause types are given; which frame a phrase needs is learned from
   them), and state what is given and what is learned. English must be unaffected or improved.
   **Check every distinct Arabic sentence the system can produce** (enumerate them from the frames
   and learned phrases) and list them in the doc with a note on each one that was changed.
2. **Calibrated "nothing unusual".** v3 learned P(no disturbance | monitor silent) = 0.957 from 11/15
   silent clean and 0/60 silent disturbed teaching episodes; on new seeds the monitor was silent in
   2/40 disturbed episodes, and the agent then said "no unusual cause was detected" for a real
   disturbance. Make this claim calibrated:
   - estimate the silent-when-disturbed rate per disturbance type with a proper prior and enough
     teaching data (generate more teaching episodes on new seeds if needed, e.g. a calibration set
     of at least 20 episodes per condition), so the probability reflects how often a disturbance goes
     unnoticed;
   - say "nothing unusual" with a strength that matches the evidence: confident when the probability
     is high, hedged otherwise (e.g. "I noticed nothing unusual, though a small disturbance could go
     unnoticed" / "لم ألاحظ شيئًا غير معتاد، لكن اضطرابًا صغيرًا قد يمر دون أن ألاحظه"), in both
     languages, with the hedge phrases taught like the other phrases;
   - measure calibration: reliability (stated confidence vs observed accuracy, binned), Brier score,
     and precision of confident "nothing unusual" claims.

## 3. Rules

- **No ground truth in the agent's words.** Event logs and disturbance parameters only score.
- **Fresh evaluation.** Seeds 68000-68009 have been seen. Use new seeds for teaching/calibration
  (e.g. 70000+) and a final, frozen evaluation on seeds never used before (e.g. **69000-69019**, with a
  new disturbance base, e.g. 80000). Report development vs final seeds.
- **Compare with v3 paired** on the same final episodes (cause correct, coverage, precision of
  issued reports, calibration), with a sign test as in `_v3_paired.py`.
- **Stay in your files:** `pai/language/*`, new `scripts/language_*.py`, `tests/test_language*.py`,
  `docs/LANGUAGE_STAGE6.md` (add a "V4" section; keep existing content), new `results/language_*`
  files. Do not change other stages (the arm, cause inference, the monitor).
- **Keep passing:** `tests/test_language*.py`, `tests/test_slice.py`, `tests/test_cognition.py`.
- Results files must not contain absolute local paths (use paths relative to the repository).

## 4. Deliverables

Code and fast unit tests (including a test that every producible Arabic `none`/hedge sentence uses
the verb-clause complementizer form); the V4 section (design, learned vs given, the full list of
Arabic sentences with corrections, calibration tables and a reliability table, paired comparison
with v3, unedited example outputs in both languages next to the truth, failures, seeds, exact
reproduce commands); `results/language_causes_v4_*.json` with per-episode records.
