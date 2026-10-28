"""Teacher-only annotations; no semantic identifiers are sent to the learner."""
import numpy as np
from pai.language.causal_teacher import PHRASES, teaching_rows

QUESTION_ROWS={
'english': {
 'why':(['Why did this happen?','Explain the cause of this episode.','What caused the problem?'],
        ['Explain why this problem happened.','What was the cause of this event?']),
 'drop':(['Why did you drop the block?','Explain why the object fell.','What caused you to lose your grip?'],
         ['Why did the block fall?','Explain the dropped object.']),
 'cf:remove_push':(['What would happen without the push?','Imagine no push: what changes?',
                    'Without an external push, would the goal change?'],
                   ['What happens if there is no push?','Would the goal change without a push?']),
 'cf:open_early':(['What if your fingers opened early?','Imagine opening the gripper before carrying.',
                   'Would early opening of the hand prevent lifting?'],
                  ['What happens if the gripper opens early?','Would opening your fingers before lifting change things?']),
 None:(['Did it drop or get pushed?','Are you asking about a push or early opening?',
         'Is this about a dropped object or early fingers?'],['Explain why you dropped it without a push.','Drop or open early?',
           'What is the weather tomorrow?','Tell me a joke.'])},
'arabic': {
 'why':(['لماذا حدث هذا؟','اشرح سبب هذه الحلقة.','ما سبب المشكلة؟'],
        ['اشرح لماذا حدثت المشكلة.','ما سبب هذا الحدث؟']),
 'drop':(['لماذا أسقطت المكعب؟','اشرح لماذا سقط الجسم.','ما سبب فقدان الإمساك بالجسم؟'],
         ['لماذا سقط المكعب؟','اشرح سقوط الجسم.']),
 'cf:remove_push':(['ماذا يحدث دون الدفعة؟','تخيل غياب الدفعة: ما الذي يتغير؟',
                    'دون دفعة خارجية هل يتغير الهدف؟'],
                   ['هل يتغير الهدف دون الدفعة؟','ماذا يحدث مع غياب الدفعة الخارجية؟']),
 'cf:open_early':(['ماذا لو فتحت أصابعك مبكرا؟','تخيل فتح الممسك قبل الحمل.',
                   'هل يمنع فتح اليد مبكرا رفع الجسم؟'],
                  ['ماذا يحدث لو فتح الممسك مبكرا؟','هل يتغير الرفع مع فتح الأصابع قبل الحمل؟']),
 None:(['هل سقط الجسم أم تلقى دفعة؟','هل تسأل عن دفعة أم فتح مبكر؟','هل الموضوع سقوط جسم أم أصابع مفتوحة مبكرا؟'],
       ['اشرح سقوط الجسم دون الدفعة.','سقوط أم فتح مبكر؟','كيف سيكون الطقس غدا؟','قل لي نكتة.'])}}


def question_rows(language):
    table=QUESTION_ROWS[language]
    return ([(text,intent) for intent,(train,_) in table.items() for text in train],
            [(text,intent) for intent,(_,test) in table.items() for text in test])


def truth_cause(row):
    events=row['truth']['disturbances']
    if not events:
        return 'none'
    return dict(payload='heavier_object',friction='slippery_object').get(events[0]['disturbance'],events[0]['disturbance'])


def make_pairs(rows, language, coordinates, *, corruption=0., seed=72801):
    """The learner receives no true cause key, parameter key, or source condition."""
    rng=np.random.default_rng(seed)
    phrases=[PHRASES[language]['cause:'+c] for c in ('none','push','heavier_object','slippery_object','camera_shift')]
    corrupt=set(rng.choice(len(rows),int(round(len(rows)*corruption)),replace=False).tolist())
    pairs=[]
    for i,row in enumerate(rows):
        phrase=PHRASES[language]['cause:'+truth_cause(row)]
        if i in corrupt:
            phrase=str(rng.choice([p for p in phrases if p!=phrase]))
        posterior,features=coordinates.encode(row['belief'])
        numbers=[]
        for event in row['truth']['disturbances']:
            prm=event['params']
            if 'mass' in prm:
                numbers.append(dict(template=PHRASES[language]['estimate:extra_mass_kg'],value=[prm['mass']]))
            if 'offset' in prm:
                numbers.append(dict(template=PHRASES[language]['estimate:camera_offset_mm'],value=(1000*np.array(prm['offset'])).tolist()))
            if event['disturbance'] in ('push','camera_shift','friction'):
                numbers.append(dict(template=PHRASES[language]['estimate:onset'],value=[event['t']*.02/.1]))
        pairs.append(dict(phrase=phrase,posterior=posterior,features=features,numbers=numbers))
    return pairs


def scaffold(language):
    return [r for r in teaching_rows(language) if not r['meaning'].startswith(('cause:','estimate:','q:'))]


# Kept out of question training and model-selection diagnostics. Evaluate only
# after freezing the checkpoint; each wording is counted once in the wording CI.
FINAL_QUESTIONS={
'english':[
 ('Can you explain the cause of what happened?','why'),
 ("What caused this episode's problem?",'why'),
 ('What caused the block to drop?','drop'),
 ('Why did your grip on the object fail?','drop'),
 ('Imagine this episode without an external push.','cf:remove_push'),
 ('Would this goal change if the push were absent?','cf:remove_push'),
 ('Would opening the gripper before lifting prevent the goal?','cf:open_early'),
 ('Imagine your fingers opening early during the carry.','cf:open_early'),
 ('Explain the push and the dropped object together.',None),
 ('Was it early opening or a push?',None),
 ('How many stars are in the sky?',None),('Please sing a song.',None)],
'arabic':[
 ('هل تشرح سبب ما حدث؟','why'),('ما الذي سبب مشكلة هذه الحلقة؟','why'),
 ('ما سبب سقوط المكعب؟','drop'),('لماذا فشل إمساكك بالجسم؟','drop'),
 ('تخيل هذه الحلقة دون دفعة خارجية.','cf:remove_push'),
 ('هل يتغير هذا الهدف لو غابت الدفعة؟','cf:remove_push'),
 ('هل يمنع فتح الممسك قبل الرفع تحقق الهدف؟','cf:open_early'),
 ('تخيل فتح أصابعك مبكرا أثناء الحمل.','cf:open_early'),
 ('اشرح الدفعة وسقوط الجسم معا.',None),('هل حدث فتح مبكر أم دفعة؟',None),
 ('كم نجمة في السماء؟',None),('من فضلك غن أغنية.',None)]}
