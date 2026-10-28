"""Teacher-owned surface semantics used only after explanations are produced."""
from pai.language.causal_v4_teacher import teacher
from pai.language.causal_teacher import PHRASES
from language_causes_v2_eval import score_v2


def score_v4(output,truth,language):
    t=teacher(language)
    decoder={new:old for old,new in t['phrases'].items()}
    decoder[t['hedge']]=PHRASES[language]['cause:none']
    # Grade the actual rendered phrase, not the inherited association's old
    # spelling. A hedge is still a none hypothesis and can be wrong.
    surface=output.get('surface_phrase')
    spoken=decoder.get(surface)
    if surface is not None and surface not in output['text']:
        spoken=None
    normalized=dict(output,phrase=spoken)
    return score_v2(normalized,truth,language)
