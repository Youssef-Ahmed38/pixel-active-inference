"""Teacher-owned dictionaries. These are given supervision, never runtime evidence."""

PHRASES = {
 'english': {
  'cause:none': 'no unusual cause was detected', 'cause:push': 'an external force pushed my arm',
  'cause:heavier_object': 'the object weighed more than expected',
  'cause:slippery_object': 'the object slipped relative to my fingers',
  'cause:camera_shift': 'the camera moved',
  'hypothesis': 'My evidence suggests that {cause}.',
  'uncertain': 'I do not have enough evidence to identify the cause.',
  'estimate:extra_mass_kg': 'I estimated about {value} kg of extra mass.',
  'estimate:camera_offset_mm': 'I estimated camera displacement (x, y, z) of about {value} mm.',
  'estimate:onset': 'I estimated onset at planning step {value}.',
  'estimate:sink_mm': 'I estimated downward slip of about {value} mm.',
  'action:recalibrated the camera': 'I recalibrated my camera.',
  'action:expect the extra weight': 'I updated my expected weight.',
  'action:carry it low, grasped at its centre': 'I changed to a low carry with a central grasp.',
  'action:carry at 50% speed': 'I reduced carrying speed to half.',
  'fallback': 'I lost my hold and returned to grasping {value} time(s).',
  'goal:yes': 'My final observation indicates that the goal was reached.',
  'goal:no': 'My final observation does not show the goal reached.',
  'goal:unknown': 'I lack a final goal observation.',
  'q:why': 'Why did things go wrong?', 'q:drop': 'Why did you drop the block?',
  'q:cf:remove_push': 'What would have happened without the push?',
  'q:cf:open_early': 'What if you had opened your fingers early?',
  'cf:unknown': 'I cannot answer this counterfactual with my available model evidence.',
  'drop:unknown': 'I cannot establish the cause of a drop from an episode-level cause estimate.',
  'cf:necessary': 'In my local model replay, this change prevented the tested subgoal. This is a prediction, not an observed outcome.',
  'cf:unnecessary': 'In my local model replay, this change still allowed the tested subgoal. This is a prediction, not an observed outcome.'},
 'arabic': {
  'cause:none': 'لم أكتشف سببا غير معتاد', 'cause:push': 'قوة خارجية دفعت ذراعي',
  'cause:heavier_object': 'الجسم كان أثقل مما توقعت', 'cause:slippery_object': 'الجسم انزلق بالنسبة إلى أصابعي',
  'cause:camera_shift': 'الكاميرا تحركت', 'hypothesis': 'تشير أدلتي إلى أن {cause}.',
  'uncertain': 'ليس لدي دليل كاف لتحديد السبب.',
  'estimate:extra_mass_kg': 'قدرت الكتلة الإضافية بحوالي {value} كغ.',
  'estimate:camera_offset_mm': 'قدرت إزاحة الكاميرا (س، ص، ع) بحوالي {value} مم.',
  'estimate:onset': 'قدرت البداية عند خطوة التخطيط {value}.',
  'estimate:sink_mm': 'قدرت الانزلاق إلى أسفل بحوالي {value} مم.',
  'action:recalibrated the camera': 'أعدت معايرة الكاميرا.',
  'action:expect the extra weight': 'حدثت توقعاتي للوزن.',
  'action:carry it low, grasped at its centre': 'غيرت الحمل ليكون منخفضا مع الإمساك من المركز.',
  'action:carry at 50% speed': 'خفضت سرعة الحمل إلى النصف.',
  'fallback': 'فقدت الإمساك وعدت إلى محاولة الإمساك {value} مرة.',
  'goal:yes': 'تشير ملاحظتي الأخيرة إلى تحقق الهدف.',
  'goal:no': 'لا تظهر ملاحظتي الأخيرة تحقق الهدف.', 'goal:unknown': 'ليس لدي رصد نهائي للهدف.',
  'q:why': 'لماذا ساءت الأمور؟', 'q:drop': 'لماذا أسقطت المكعب؟',
  'q:cf:remove_push': 'ماذا كان سيحدث دون الدفعة؟',
  'q:cf:open_early': 'ماذا لو فتحت أصابعك مبكرا؟',
  'cf:unknown': 'لا أستطيع الإجابة عن هذا الافتراض بالأدلة المتاحة من نموذجي.',
  'drop:unknown': 'لا أستطيع إثبات سبب السقوط من تقدير سبب للحلقة بأكملها.',
  'cf:necessary': 'في إعادة المحاكاة المحلية بنموذجي منع هذا التغيير الهدف الفرعي المختبر. هذا تنبؤ وليس نتيجة مرصودة.',
  'cf:unnecessary': 'في إعادة المحاكاة المحلية بنموذجي ظل الهدف الفرعي المختبر ممكنا مع هذا التغيير. هذا تنبؤ وليس نتيجة مرصودة.'}}


def teaching_rows(language):
    rows = []
    for key, template in PHRASES[language].items():
        slots = ['value'] if '{value}' in template else ['cause'] if '{cause}' in template else []
        for value in ('0.2', '0.5') if slots == ['value'] else ('alpha', 'beta') if slots else ('',):
            values = {k: value for k in slots}
            rows.append(dict(meaning=key, slots=values, text=template.format(**values)))
    return rows


def episode_teaching_rows(language, episodes):
    """Pair observed teaching episodes with teacher cause labels and clauses.

    Ground-truth labels are permitted supervision here, never inference input.
    All five labels must actually occur; absent causes remain untaught.
    """
    rows = [r for r in teaching_rows(language) if not r['meaning'].startswith('cause:')]
    mapping = {'payload':'heavier_object', 'friction':'slippery_object'}
    for episode in episodes:
        if episode['mode'] != 'baseline':
            continue
        events = episode['truth']['disturbances']
        labels = {mapping.get(e['disturbance'], e['disturbance']) for e in events} or {'none'}
        for label in labels:
            key = 'cause:' + label
            rows.append(dict(meaning=key, text=PHRASES[language][key], slots={},
                             source_seed=episode['seed'], source_condition=episode['condition'],
                             observation=episode['belief']))
    # New episode numbers at evaluation test slot generalization, not new grammar.
    for key in ('why','drop','cf:remove_push','cf:open_early'):
        prefix = 'In episode {episode}: ' if language == 'english' else 'في الحلقة {episode}: '
        for number in (11,22):
            rows.append(dict(meaning='q:'+key+':episode', slots={'episode':str(number)},
                             text=prefix.format(episode=number)+PHRASES[language]['q:'+key]))
    return rows
