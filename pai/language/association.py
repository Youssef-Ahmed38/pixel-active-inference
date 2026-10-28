"""Cross-situational associations over anonymous belief coordinates.

Learners see vectors and utterances, never simulator labels or arm variable names.
The boundary adapter's coordinate permutation is given, not a learned dictionary.
"""
import math
import re
from collections import Counter
import numpy as np

from pai.language.causal import CausalExplainer, ClauseLanguage, EpisodeEvidence


def words(text):
    return re.findall(r'[^\W\d_]+',text.casefold(),flags=re.UNICODE)


class BeliefCoordinates:
    def __init__(self, causes, features):
        self.causes, self.features = list(causes), [tuple(f) for f in features]

    @classmethod
    def fit(cls, beliefs, seed=72800):
        rng=np.random.default_rng(seed)
        causes=sorted({c for b in beliefs for c in b['posterior']})
        features=sorted({(c,k) for b in beliefs for c,params in b['parameters'].items()
                         for k,v in params.items() if isinstance(v,(int,float,list))})
        rng.shuffle(causes)
        rng.shuffle(features)
        return cls(causes,features)

    def encode(self, belief):
        p=np.array([belief['posterior'].get(c,0.) for c in self.causes],float)
        features=[]
        for c,k in self.features:
            value=belief['parameters'].get(c,{}).get(k)
            vector=np.asarray(value if isinstance(value,list) else [value],dtype=float)
            features.append(vector.tolist() if np.isfinite(vector).all() else None)
        return p.tolist(),features


class EvidenceLexicon:
    def __init__(self, phrase_by_coordinate, numeric, diagnostics):
        self.phrase_by_coordinate=phrase_by_coordinate
        self.numeric=numeric
        self.diagnostics=diagnostics

    @classmethod
    def fit(cls, pairs, *, min_support=1.5):
        if not pairs:
            return cls({},[],{'episodes':0})
        matrix=np.asarray([r['posterior'] for r in pairs],float)
        if (matrix.ndim!=2 or not np.isfinite(matrix).all() or (matrix<0).any()
            or np.any((matrix.sum(1)>0)&(~np.isclose(matrix.sum(1),1.)))):
            raise ValueError('Anonymous posteriors must be normalized or absent')
        # Expected co-occurrence mass, normalized by the marginal cause rate.
        prior=matrix.mean(0)
        mappings,details={},{}
        for phrase in sorted({r['phrase'] for r in pairs}):
            mask=np.array([r['phrase']==phrase for r in pairs])
            mass=matrix[mask].sum(0)
            pmi=np.log((mass+1e-6)/(max(1,int(mask.sum()))*prior+1e-6))
            # Weight PMI by conditional posterior mass: a rare catch-all that
            # appears once must not tie a specific variable supported repeatedly.
            association=pmi*mass/max(1,int(mask.sum()))
            ranking=np.argsort(association)[::-1]
            best=int(ranking[0])
            margin=float(association[best]-association[ranking[1]]) if len(ranking)>1 else float(association[best])
            usable=bool(mass[best]>=min_support and pmi[best]>0 and margin>.1)
            details[phrase]=dict(coordinate=best,support=float(mass[best]),pmi=pmi.tolist(),
                                 margin=margin,usable=usable)
            if usable:
                mappings.setdefault(str(best),[]).append(phrase)
        # Competing phrases for one latent variable are not silently resolved.
        mappings={k:v[0] for k,v in mappings.items() if len(v)==1}
        numeric=[]
        for phrase,template in sorted({(r['phrase'],q['template']) for r in pairs for q in r['numbers']}):
            group=[(r,q['value']) for r in pairs if r['phrase']==phrase
                   for q in r['numbers'] if q['template']==template]
            candidates=[]
            for j in range(len(pairs[0]['features'])):
                matched=[(r['features'][j],target) for r,target in group
                         if r['features'][j] is not None and len(r['features'][j])==len(target)]
                if len(matched)<3:
                    continue
                x,y=map(np.asarray,zip(*matched))
                # Numbers are taught in stated common units. No fitted scale may
                # make a wrong-unit feature look equivalent to the true feature.
                spread=float(np.std(y))
                if spread<1e-6 or len(np.unique(y,axis=0))<3:
                    continue
                error=float(np.sqrt(np.mean((x-y)**2))/spread)
                corr=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if np.std(x)>0 else 0.
                if math.isfinite(corr):
                    candidates.append((error,j,corr,len(matched)))
            candidates.sort()
            if not candidates:
                continue
            error,j,corr,support=candidates[0]
            margin=candidates[1][0]-error if len(candidates)>1 else 1.
            if error<.5 and corr>.5 and margin>.05:
                numeric.append(dict(phrase=phrase,template=template,feature=j,
                                    normalized_rmse=error,correlation=corr,support=support,margin=margin))
        return cls(mappings,numeric,dict(episodes=len(pairs),phrases=details))


class QuestionIntent:
    """Independent TF-IDF nearest-example intent induction; no phrase rules."""
    def __init__(self, vocabulary, idf, examples, threshold=.25, margin=.08):
        self.vocabulary=vocabulary
        self.idf=np.asarray(idf)
        self.examples=examples
        self.threshold,self.margin=threshold,margin

    def vector(self,text):
        counts=Counter(words(text))
        v=np.array([counts.get(w,0) for w in self.vocabulary])*self.idf
        return v/max(float(np.linalg.norm(v)),1e-12)

    @classmethod
    def fit(cls, rows):
        vocab=sorted({w for text,_ in rows for w in words(text)})
        idf=[math.log((1+len(rows))/(1+sum(w in words(t) for t,_ in rows)))+1 for w in vocab]
        model=cls(vocab,idf,[])
        model.examples=[(intent,model.vector(text).tolist()) for text,intent in rows]
        return model

    def predict(self,text):
        v=self.vector(text)
        scores={}
        for intent,example in self.examples:
            scores[intent]=max(scores.get(intent,0.),float(v@np.asarray(example)))
        ranked=sorted(scores.items(),key=lambda r:r[1],reverse=True)
        if not ranked or ranked[0][1]<self.threshold or ranked[0][1]-ranked[1][1]<self.margin:
            return None
        return ranked[0][0]


class AssociatedLanguage(ClauseLanguage):
    def __init__(self, templates, questions):
        super().__init__(templates)
        self.questions=questions

    def question(self,text,episode=None):
        numbers=re.findall(r'\d+',text)
        if any(int(n)!=episode for n in numbers):
            return None
        return self.questions.predict(text)


class AssociatedExplainer:
    def __init__(self, coordinates, lexicon, templates, questions, threshold=.9):
        self.coordinates,self.lexicon=coordinates,lexicon
        self.templates,self.questions,self.threshold=templates,questions,threshold
        runtime=dict(templates)
        for coordinate,phrase in lexicon.phrase_by_coordinate.items():
            runtime['cause:'+coordinates.causes[int(coordinate)]]=phrase
        self.language=AssociatedLanguage(runtime,questions)
        self.base=CausalExplainer(self.language,threshold)

    def explain(self,evidence):
        # The old numeric-name dictionary is deliberately not used here.
        safe=EpisodeEvidence(evidence.episode,evidence.posterior,{},evidence.adaptations,
                             evidence.fallbacks,evidence.goal_observed,evidence.credits)
        out=self.base.explain(safe)
        out['agent_cause']=out.pop('cause')
        out['phrase']=None
        out['numeric_claims']=[]
        if out['agent_cause'] is not None:
            index=self.coordinates.causes.index(out['agent_cause'])
            phrase=self.lexicon.phrase_by_coordinate[str(index)]
            out['phrase']=phrase
            _,features=self.coordinates.encode(dict(posterior=evidence.posterior,parameters=evidence.parameters))
            for mapping in self.lexicon.numeric:
                if mapping['phrase']!=phrase:
                    continue
                value=features[mapping['feature']]
                if value is None:
                    continue
                text=mapping['template'].format(value=', '.join(f'{v:g}' for v in value))
                out['text']+=' '+text
                out['numeric_claims'].append(dict(template=mapping['template'],value=value,
                                                  feature=mapping['feature'],text=text))
        return out

    def answer(self,text,evidence):
        intent=self.language.question(text,evidence.episode)
        if intent=='why':
            return self.explain(evidence)
        return self.base.answer(text,evidence)
