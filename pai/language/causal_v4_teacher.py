"""Teacher-owned grammatical supervision; no clause-type labels reach the learner."""
from pai.language.causal_teacher import PHRASES

ARABIC_EMBEDDINGS = [
 ('لم أكتشف سببًا غير معتاد', 'تشير أدلتي إلى أنني لم أكتشف سببًا غير معتاد.'),
 ('لم ألاحظ شيئًا غير معتاد، لكنّ اضطرابًا صغيرًا قد يمر دون أن ألاحظه',
  'تشير أدلتي إلى أنني لم ألاحظ شيئًا غير معتاد، لكنّ اضطرابًا صغيرًا قد يمر دون أن ألاحظه.'),
 ('قوةً خارجيةً دفعت ذراعي', 'تشير أدلتي إلى أنّ قوةً خارجيةً دفعت ذراعي.'),
 ('الجسم كان أثقل مما توقعت', 'تشير أدلتي إلى أنّ الجسم كان أثقل مما توقعت.'),
 ('الجسم انزلق بالنسبة إلى أصابعي', 'تشير أدلتي إلى أنّ الجسم انزلق بالنسبة إلى أصابعي.'),
 ('الكاميرا تحركت', 'تشير أدلتي إلى أنّ الكاميرا تحركت.'),
 ('تحركت', 'تشير أدلتي إلى أنها تحركت.'),
 ('تحرك', 'تشير أدلتي إلى أنه تحرك.'),
 ('الكتلة الإضافية جعلت الجسم أثقل', 'تشير أدلتي إلى أنّ الكتلة الإضافية جعلت الجسم أثقل.'),
 ('الهدف تحقق', 'تشير أدلتي إلى أنّ الهدف تحقق.'),
 ('المهمة تحققت', 'تشير أدلتي إلى أنّ المهمة تحققت.'),
 ('الأجسام الأثقل تحركت', 'تشير أدلتي إلى أنّ الأجسام الأثقل تحركت.'),
]

def teacher(language):
    if language == 'english':
        phrases = {PHRASES[language]['cause:'+c]: PHRASES[language]['cause:'+c]
                   for c in ('none','push','heavier_object','slippery_object','camera_shift')}
        hedge = 'I noticed nothing unusual, though a small disturbance could go unnoticed'
        pairs = [(p, 'My evidence suggests that '+p+'.') for p in [*phrases.values(),hedge]]
        return dict(phrases=phrases,hedge=hedge,pairs=pairs,replacements={})
    old=PHRASES['arabic']
    replacements={
      old['uncertain']: 'ليس لدي دليل كافٍ لتحديد السبب.',
      old['cf:unknown']: 'لا أستطيع الإجابة عن هذا السؤال الافتراضي بالأدلة المتاحة من نموذجي.',
      old['drop:unknown']: 'لا أستطيع تحديد سبب سقوط الجسم اعتمادًا على تقدير سبب الاضطراب في الحلقة بأكملها.',
      old['action:expect the extra weight']: 'حدّثت توقعاتي بشأن الوزن.',
      old['action:carry it low, grasped at its centre']: 'غيّرت أسلوب الحمل إلى حمل الجسم على ارتفاع منخفض مع الإمساك به من مركزه.',
      old['fallback']: 'فقدت الإمساك بالجسم. عدد مرات إعادة محاولة الإمساك: {value}.',
      old['cf:necessary']: 'في إعادة المحاكاة المحلية بنموذجي، منع هذا التغيير تحقق الهدف الفرعي المختبَر. هذا تنبؤ وليس نتيجة مرصودة.',
      old['cf:unnecessary']: 'في إعادة المحاكاة المحلية بنموذجي، ظل تحقق الهدف الفرعي المختبَر ممكنًا مع هذا التغيير. هذا تنبؤ وليس نتيجة مرصودة.',
    }
    phrases={old['cause:'+c]:ARABIC_EMBEDDINGS[i][0] for c,i in
             [('none',0),('push',2),('heavier_object',3),('slippery_object',4),('camera_shift',5)]}
    return dict(phrases=phrases,hedge=ARABIC_EMBEDDINGS[1][0],pairs=ARABIC_EMBEDDINGS,replacements=replacements)

def probability_rows(language):
    template = ('I estimate the probability of no disturbance at about {value}%.' if language=='english'
                else 'أقدّر احتمال عدم وجود اضطراب بنحو {value}٪.')
    return [dict(meaning='probability',slots={'value':str(v)},text=template.format(value=v)) for v in (65,85)]
