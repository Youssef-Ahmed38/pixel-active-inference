"""Template conversation teacher; word meanings are not available at inference."""
from dataclasses import replace
import itertools

from pai.language.grounding import GroundedFrame


VOCABULARY = {
    "english": {"red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
                "block": "block", "cup": "cup", "key": "key", "plate": "plate", "bowl": "bowl", "box": "box",
                "on": "on", "in": "inside", "near": "near", "fell": "fell", "stalled": "stalled", "slipped": "slipped",
                "slippery": "slippery", "blocked": "blocked", "heavy": "heavy", "yes": "yes", "no": "no"},
    "arabic": {"red": "أحمر", "blue": "أزرق", "green": "أخضر", "yellow": "أصفر",
               "block": "مكعب", "cup": "كوب", "key": "مفتاح", "plate": "طبق", "bowl": "وعاء", "box": "صندوق",
               "on": "فوق", "in": "داخل", "near": "قرب", "fell": "سقط", "stalled": "توقف", "slipped": "انزلق",
               "slippery": "زلق", "blocked": "مسدود", "heavy": "ثقيل", "yes": "نعم", "no": "لا"}}


def utterance(frame, language="english"):
    words = VOCABULARY[language]
    s = f"{words[frame.subject_color]} {words[frame.subject_kind]}"
    t = None if frame.target_kind is None else f"{words[frame.target_color]} {words[frame.target_kind]}"
    p = words[frame.predicate]
    if language == "english":
        if frame.intent == "instruction": return f"put the {s} {p} the {t}"
        if frame.intent == "question": return f"is the {s} {p} the {t}"
        if frame.intent == "answer": return f"{words[frame.truth]} the {s} is {'not ' if frame.truth == 'no' else ''}{p} the {t}"
        if frame.intent == "explanation": return f"the {s} {p} because it was {words[frame.cause]}"
        if frame.intent == "hypothesis": return f"the {s} {p} possibly because it was {words[frame.cause]}"
        return f"the {s} is {p} the {t}" if t else f"the {s} {p}"
    s = f"{words[frame.subject_kind]} {words[frame.subject_color]}"
    t = None if frame.target_kind is None else f"{words[frame.target_kind]} {words[frame.target_color]}"
    if frame.intent == "instruction": return f"ضع {s} {p} {t}"
    if frame.intent == "question": return f"هل {s} {p} {t}"
    if frame.intent == "answer": return f"{words[frame.truth]} {s} {'ليس ' if frame.truth == 'no' else ''}{p} {t}"
    if frame.intent == "explanation": return f"{s} {p} لأنه كان {words[frame.cause]}"
    if frame.intent == "hypothesis": return f"{s} {p} ربما لأنه كان {words[frame.cause]}"
    return f"{s} {p} {t}" if t else f"{s} {p}"


def curriculum(language="english", include_hypotheses=False):
    rows = []
    colors = ("red", "blue", "green", "yellow")
    for sc, sk, tc, tk, predicate in itertools.product(colors, ("block", "cup", "key"), colors,
                                                      ("plate", "bowl", "box"), ("on", "in", "near")):
        for intent in ("instruction", "description", "question"):
            frame = GroundedFrame(intent, predicate, sc, sk, tc, tk)
            rows.append((utterance(frame, language), frame))
        for truth in ("yes", "no"):
            frame = GroundedFrame("answer", predicate, sc, sk, tc, tk, truth=truth)
            rows.append((utterance(frame, language), frame))
    for sc, sk, predicate in itertools.product(colors, ("block", "cup", "key"), ("fell", "stalled", "slipped")):
        frame = GroundedFrame("description", predicate, sc, sk)
        rows.append((utterance(frame, language), frame))
        for cause in ("slippery", "blocked", "heavy"):
            frame = replace(frame, intent="explanation", cause=cause)
            rows.append((utterance(frame, language), frame))
            if include_hypotheses:
                hypothesis = replace(frame, intent="hypothesis")
                rows.append((utterance(hypothesis, language), hypothesis))
    # Entire color/predicate/target-kind combinations are held out across intents.
    train, test = [], []
    for text, frame in rows:
        key = colors.index(frame.subject_color) + (0 if frame.target_kind is None else ("plate", "bowl", "box").index(frame.target_kind))
        key += ("on", "in", "near", "fell", "stalled", "slipped").index(frame.predicate)
        (test if key % 5 == 0 else train).append((text, frame))
    return train, test
