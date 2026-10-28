"""Evidence-only V4 composition and per-condition Bayesian silence calibration."""
import math
import numpy as np
from pai.language.association import AssociatedExplainer, BeliefCoordinates, EvidenceLexicon, QuestionIntent

class LearnedComposition:
    def __init__(self, frames, choices):
        self.frames,self.choices=frames,choices

    @classmethod
    def fit(cls,pairs):
        frames,choices=[],{}
        for phrase,sentence in pairs:
            if sentence.count(phrase)!=1:
                raise ValueError('Teacher must embed a phrase exactly once')
            frame=sentence.replace(phrase,'{clause}')
            if frame not in frames:
                frames.append(frame)
            choice=frames.index(frame)
            if phrase in choices and choices[phrase]!=choice:
                raise ValueError('Conflicting frame teaching')
            choices[phrase]=choice
        return cls(frames,choices)

    def say(self,phrase):
        # Unknown compositions abstain, rather than guessing Arabic morphology.
        if phrase not in self.choices:
            return None
        return self.frames[self.choices[phrase]].format(clause=phrase)

class CalibratedSilence:
    def __init__(self, counts, alpha=.5):
        self.counts,self.alpha=counts,alpha

    @classmethod
    def fit(cls,rows,cause_of,alpha=.5):
        counts={}
        for r in rows:
            c=cause_of(r)
            item=counts.setdefault(c,dict(n=0,silent=0,explicit_none=0))
            item['n']+=1
            p=r['belief']['posterior']
            item['silent']+=int(not p)
            item['explicit_none']+=int(bool(p) and max(p,key=p.get)=='none')
        if 'none' not in counts or len(counts)<2:
            raise ValueError('Need clean and disturbed teaching episodes')
        return cls(counts,alpha)

    def posterior(self,event='silent'):
        # A balanced experimental prior is explicitly given. Each cause has its
        # own Beta(alpha,alpha) likelihood; do not pool 0/80 into one rare event.
        a=self.alpha
        mass={c:(r[event]+a)/(r['n']+2*a) for c,r in self.counts.items()}
        total=sum(mass.values())
        return {c:v/total for c,v in mass.items()}

    def diagnostics(self,event='silent'):
        rng=np.random.default_rng(70420)
        draws={c:rng.beta(r[event]+self.alpha,r['n']-r[event]+self.alpha,20000)
               for c,r in self.counts.items()}
        prob=draws['none']/sum(draws.values())
        return dict(posterior=self.posterior(event),none_credible95=np.quantile(prob,[.025,.975]).tolist(),
                    likelihoods={c:(r[event]+self.alpha)/(r['n']+2*self.alpha) for c,r in self.counts.items()})

class V4Explainer:
    def __init__(self,base,composition,teacher_surface,hedge,replacements,silence,probability_frame,confident=.95,minimum=.5):
        self.base,self.composition=base,composition
        self.teacher_surface,self.hedge,self.replacements=teacher_surface,hedge,replacements
        self.silence,self.confident,self.minimum=silence,confident,minimum
        self.probability_frame=probability_frame
        self.questions,self.language=base.questions,base.language

    def explain(self,evidence):
        out=self.base.explain(evidence)
        original=self.base.language.say('hypothesis',cause=out['phrase']) if out['phrase'] else self.base.language.say('uncertain')
        valid=bool(evidence.posterior) and all(math.isfinite(v) and 0<=v<=1 for v in evidence.posterior.values()) and math.isclose(sum(evidence.posterior.values()),1.,abs_tol=1e-6)
        event='silent' if not evidence.posterior else 'explicit_none' if valid and max(evidence.posterior,key=evidence.posterior.get)=='none' else None
        if event:
            p=self.silence.posterior(event)['none']
            issued=p>=self.minimum
            out.update(agent_cause='none' if issued else None, confidence=p,abstained=not issued,
                       evidence_source='silent_monitor' if event=='silent' else 'calibrated_none_monitor',
                       claim_strength='confident' if p>=self.confident else 'hedged' if issued else 'abstained',
                       numeric_claims=[],phrase=self.base.language.templates['cause:none'] if issued else None)
            out['none_probability']=p
            surface=self.hedge if p<self.confident else self.teacher_surface[out['phrase']] if issued else None
            replacement=self.composition.say(surface) if issued else self.base.language.say('uncertain')
        else:
            out['claim_strength']='confident' if not out['abstained'] else 'abstained'
            surface=self.teacher_surface.get(out['phrase'])
            replacement=self.composition.say(surface) if surface else self.base.language.say('uncertain')
        if event:
            replacement += ' '+self.probability_frame.format(value=f'{100*p:.1f}')
        if replacement is None:
            raise ValueError('Untaught composition')
        out['surface_phrase']=surface if not out['abstained'] else None
        out['text']=replacement+out['text'][len(original):]
        for before,after in self.replacements.items():
            if '{value}' not in before:
                out['text']=out['text'].replace(before,after)
        # Variable cardinalities use a taught count-label construction, avoiding
        # incorrect Arabic singular/plural/case after arbitrary counts.
        if evidence.fallbacks:
            before=self.base.language.templates['fallback']
            if before in self.replacements:
                out['text']=out['text'].replace(before.format(value=evidence.fallbacks),self.replacements[before].format(value=evidence.fallbacks))
        return out

    def answer(self,text,evidence):
        if self.language.question(text,evidence.episode)=='why':
            return self.explain(evidence)
        out=self.base.answer(text,evidence)
        for before,after in self.replacements.items():
            out['text']=out['text'].replace(before,after)
        return out

def load_v4(checkpoint):
    coord=BeliefCoordinates(**checkpoint['coordinates'])
    silence=CalibratedSilence(**checkpoint['v4']['silence'])
    out={}
    for lang,m in checkpoint['models'].items():
        base=AssociatedExplainer(coord,EvidenceLexicon(**m['lexicon']),m['templates'],QuestionIntent(**m['questions']))
        v=checkpoint['v4']['languages'][lang]
        out[lang]=V4Explainer(base,LearnedComposition(**v['composition']),v['phrases'],v['hedge'],v['replacements'],silence,v['probability_frame'])
    return out

