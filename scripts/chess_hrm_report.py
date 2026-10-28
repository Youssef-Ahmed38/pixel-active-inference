"""Append the frozen adaptive-HRM experiment to CHESS_STRENGTH.md."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.chess_hrm_study import load,PLAN,GAMES,atomic,sha


def percent(v):return f'{100*v:.1f}%'
def interval(v):return f'[{100*v[0]:.1f}, {100*v[1]:.1f}]%'
def rating(r):
    if r is None:return 'unavailable'
    if r.get('estimate') is None:return 'no finite estimate'
    lo,hi=r['ci95']
    return f"{r['estimate']:.0f} [{'-∞' if lo is None else f'{lo:.0f}'}, {'+∞' if hi is None else f'{hi:.0f}'}]"


def report():
    plan=load(PLAN);data=load(GAMES);s=load('results/chess_hrm_search_summary.json.gz')
    audit=load('results/chess_hrm_final_audit.json.gz');halt=load('results/chess_hrm_final_halting.json.gz')
    training=load('results/chess_hrm_training.json.gz');tests=load('results/chess_hrm_tests.json.gz')
    assert data['complete'] and audit['passed'] and s['schedule_complete']
    h=halt['summary'];paired=s['groups']['paired'];games=data['games']
    live=[r['halting'] for g in games for r in g['search_records'] if 'halting' in r]
    live_hist={str(k):sum(r['segments_histogram'].get(str(k),0) for r in live) for k in range(1,7)}
    live_views=sum(live_hist.values())
    live_summary={'views':live_views,'segments_histogram':live_hist,
                  'mean_segments':sum(int(k)*v for k,v in live_hist.items())/live_views if live_views else None}
    levels=[1320,1500,1700];gains={};rng=np.random.default_rng(0)
    for level in levels:
        groups={tag:[g for g in games if g['group']==f'{tag}_elo{level}'] for tag in ('baseline','adaptive')}
        from pai.chess.ladder import candidate_score
        differences=[]
        for pair in range(plan['pairs_per_ladder_rung']):
            vals={tag:[candidate_score(g) for g in rows if g['pair']==pair] for tag,rows in groups.items()}
            if all(len(v)==2 and all(x is not None for x in v) for v in vals.values()):
                differences.append(np.mean(vals['adaptive'])-np.mean(vals['baseline']))
        if differences:
            a=np.array(differences);draws=rng.integers(0,len(a),(5000,len(a)))
            gains[str(level)]={'opening_pairs':len(a),'mean_score_gain':float(a.mean()),
                               'paired_opening_bootstrap95':np.quantile(a[draws].mean(1),[.025,.975]).tolist()}
    meta={'plan_sha256':sha(PLAN),'games_sha256':sha(GAMES),'halting':h,'paired_match':paired,
          'conditional_engine_anchor_elo':s['conditional_engine_anchor_elo'],'paired_ladder_score_gains':gains,
          'audit':audit,'in_search_halting':live_summary,'promoted_manifest_changed':False,
          'paired_promotion_gate_passed':bool(paired.get('promotion_gate_passed',False))}
    # Promotion flag is evidence from this protocol only; never write a manifest.
    atomic(Path('results/chess_hrm_report.json.gz'),meta)
    lines=['\n## Adaptive HRM in search with learned halting (2026-10-28)\n',
      'This is a separate opt-in candidate. The previous promoted configuration and checkpoints remain intact. '
      'The paired match tests playing strength; lower teacher error alone is not a promotion gate. '
      'This study uses **0.2 seconds per move**, equally for both sides. Its ratings are conditional on that '
      'protocol and must not be substituted for the historical 1-second 1037 Elo estimate.\n',
      '### Design: learned versus given\n',
      '`pai/chess/adaptive_hrm.py` reuses the existing HRM encoder, coupled recurrent L/H modules, '
      'readout and two-logit halt head. Rank patches of eight board squares reduce the sequence to eight '
      'evidence tokens plus the readout. A 32-dimensional residual head refines the promoted linear value. '
      'Colour-mirrored views are evaluated independently and antisymmetrized, preserving colour symmetry. '
      'Alpha-beta calls this value at nonterminal search nodes through the existing batched, cached value interface; '
      'the same learned move prior orders moves for both players. The halting histogram in each move record counts '
      'actual executed recurrent segments, excluding cached values and fixed initial-board warm-up.\n',
      'Given: public symbolic board tokens/context, rank patching, architecture, six-segment safety cap, '
      '500-cp residual scale, search algorithm and budgets, a 2-cp cost per additional segment, and Stockfish labels. '
      'Learned: residual encoder/recurrent/readout weights and the halt classifier. The latter is supervised '
      'from when **future teacher-error reduction exceeds added recurrence cost**. Labels are detached; the backbone '
      'is frozen while fitting the halt head. No teacher error, legal oracle or engine output enters inference. '
      'This is supervised stopping, not a claim of chess reinforcement Q-learning. Existing cause HRM behavior is unchanged.\n',
      'Backbone selection uses validation full-depth error; halt-head selection uses validation error plus '
      '2 cp × mean segments. Training samples come only from the old depth-8 training corpus; train/validation '
      'board/mirror orbits are disjoint. Validation is development data previously used in chess work. '
      'Final data were not used for fitting or selection.\n',
      f"The fit retained {len(training['training_oracle_steps'])} unique teaching orbits and "
      f"{training['validation']['positions']} development validation positions, with "
      f"{len(training['value_history'])} value epochs and {len(training['halt_history'])} halt-head epochs.\n",
      '### Fresh halting analysis\n',
      f"Seed 93400 generated {h['generated']} depth-8 teacher-labelled positions; {h['positions']} survived "
      "teaching-orbit/duplicate exclusion. This positional test is separate from the opening books.\n",
      '| Measure | Frozen candidate |','|---|---:|',
      f"| Mean segments (cap six) | {h['mean_segments']:.3f} |",
      f"| Recurrent segments saved versus fixed six | {100*h['recurrence_saving_vs_cap']:.1f}% |",
      f"| Easiest baseline-error quartile | {h['easy_quartile_mean_segments']:.3f} |",
      f"| Hardest baseline-error quartile | {h['hard_quartile_mean_segments']:.3f} |",
      f"| Baseline error / steps Pearson correlation | {h['baseline_error_step_correlation']:.3f} |",
      f"| Baseline / adaptive / fixed-six MAE | {h['baseline_mae_cp']:.2f} / {h['adaptive_mae_cp']:.2f} / {h['fixed6_mae_cp']:.2f} cp |",
      f"\nExecuted-view segment histogram: `{h['segments_histogram_views']}`. Each position contributes two views. "
      "The hardest quartile is defined by baseline absolute teacher error, only for scoring. These are descriptive "
      "position statistics: trajectories are correlated, and the segment saving is not an end-to-end search speedup.\n",
      f"During the actual matches/ladder, {live_views} uncached HRM views used mean "
      f"{live_summary['mean_segments']:.3f} segments; histogram `{live_hist}`.\n",
      '### Equal-time paired match\n',
      f"Opening seed **97400**, ten roots, both colours, {paired['games']} completed games. "
      f"Adaptive candidate W/D/L: **{paired.get('wins',0)}/{paired.get('draws',0)}/{paired.get('losses',0)}**.\n",
      f"Score {percent(paired['score'])}, reported opening-pair 95% interval {interval(paired['score_ci95_reported'])}; "
      f"one-sided paired sign test p={paired['paired_sign_test_p_one_sided']:.4g}.\n",
      'The candidate fails the paired promotion gate and is **not promoted**. This negative strength result is '
      'retained despite positive halting and positional-error results.\n',
      '### Enlarged Stockfish ladder\n',
      'Opening seed **98400**, twenty new roots reused across both players and all three anchors. '
      'Each player has **40 games per rung**, twice the previous twenty at this move target. '
      'The engine has Threads 1 / Hash 16 MB / UCI_LimitStrength; no extra engine node/depth cap. '
      'Both agents retain quiescence two, PV reuse, the same prior, and the million-node safety cap.\n',
      '| Player | UCI anchor | Games | W/D/L | Score / reported pair 95% interval | Conditional anchor Elo / 95% interval |',
      '|---|---:|---:|---|---|---|']
    for tag in ('baseline','adaptive'):
        for elo in levels:
            r=s['groups'][f'{tag}_elo{elo}']
            lower,upper=r['elo_difference_ci95']
            per_rung=(f"{'no finite point' if r['elo_difference'] is None else f'{elo+r["elo_difference"]:.0f}'} "
                      f"[{'-∞' if lower is None else f'{elo+lower:.0f}'}, {'+∞' if upper is None else f'{elo+upper:.0f}'}]")
            lines.append(f"| {tag} | {elo} | {r['games']} | {r.get('wins',0)}/{r.get('draws',0)}/{r.get('losses',0)} | "
                         f"{percent(r['score'])} / {interval(r['score_ci95_reported'])} | {per_rung} |")
    lines += ['\n| Player | Overall conditional engine-anchor Elo / 95% interval |','|---|---|']
    for tag,r in s['conditional_engine_anchor_elo'].items():lines.append(f'| {tag} | {rating(r)} |')
    lines += ['\nBoth point estimates are equal because the players earned the same total ladder points under '
              'the same anchor exposures. The distribution over openings differs, producing different bootstrap '
              'intervals. This noisy ladder result does not overturn the negative direct paired match.\n',
              'These use the existing anchored-score equation and opening-cluster bootstrap, preserving both '
              'colours and cross-rung dependencies. Degenerate per-rung bootstraps use the existing conservative '
              'pair Hoeffding bounds; null Elo endpoints mean unbounded. Configured anchors are not human-calibrated ratings.\n',
              '| Player / anchor | Mean nodes | Mean main depth | Mean search seconds | Maximum search seconds |',
              '|---|---:|---:|---:|---:|']
    for tag in ('baseline','adaptive'):
        for elo in levels:
            name=f'{tag}_elo{elo}';r=s['groups'][name]
            depth=np.mean([t['depth'] for g in games if g['group']==name for t in g['search_records']])
            lines.append(f"| {tag} / {elo} | {r['mean_nodes']:.1f} | {depth:.3f} | {r['mean_search_seconds']:.3f} | {r['maximum_search_seconds']:.3f} |")
    lines += ['\n### Failures, scope, and verification\n',
              f"All {audit['games']} scheduled PGNs were replayed independently for legal moves, results, and draw-claim witnesses. "
              f"Unresolved games by group: `{ {k:v['unresolved'] for k,v in audit['groups'].items()} }`.\n",
              'The models are frozen; a time-based game cannot be replayed bit-for-bit. Soft deadlines can overrun '
              'during model operations. CPU contention and limited independent opening clusters remain limitations. '
              'A 1024-ply cap is an execution ceiling, never draw adjudication. No failure or unfinished game is silently '
              'converted to a score. Positive halt/error results do not establish a playing-strength gain. '
              'No promoted manifest was changed, regardless of this candidate’s result.\n',
              'Main depth excludes quiescence. Halt targets optimize positional teacher error plus recurrent cost, '
              'not search win rate. Selective reasoning or a learned move-ordering head would need a new development '
              'study; this completed final set must not become its tuning set. No 1-second HRM ladder was run.\n',
              'Restricted Windows named pipes prevented python-chess asynchronous UCI and multiprocessing queues. '
              'The new scripts explicitly opt into `uci_sync.py` (ordinary pipes, bounded UCI limits, history preserved) '
              'and four separate worker processes with atomic gzip file checkpoints. The learned player never imports '
              'that transport. Per-worker locks prevent duplicate concurrent writers. Restarting adopts completed games; '
              'frozen code/checkpoint mismatches are rejected. No user/cache absolute paths are written to results.\n',
              f"Regression: {len(tests['tests'])} tests passed in the full required cohort; six focused adaptive tests also passed "
              "(the latter include one new UCI protocol test added after full collection). Tests exercise true halting compute, "
              "mirror symmetry, save/load, benefit labels, and an actual agent move with teacher imports and engine startup blocked.\n",
              '### Seeds and exact commands\n',
              'Training/optimizer 91400; source teacher data 10400–10403 and development validation 20400–20401. '
              'Capped smoke 95400. Final halting positions 93400; paired openings 97400; ladder openings 98400. '
              'Old published opening books and teaching orbits are excluded during book construction.\n',
              '```powershell',"$env:PYTHONPATH='.'","$env:PYTHONHASHSEED='0'","$env:OMP_NUM_THREADS='1'","$env:OPENBLAS_NUM_THREADS='1'",
              '.venv/Scripts/python.exe scripts/chess_hrm_train.py',
              '.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py tests',
              '.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py smoke',
              '.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py halting',
              '.venv/Scripts/python.exe scripts/chess_hrm_study.py register --seconds .2 --pairs 20',
              '.venv/Scripts/python.exe scripts/chess_hrm_study.py run --workers 4',
              '.venv/Scripts/python.exe scripts/chess_hrm_study.py analyse',
              '.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py audit',
              '.venv/Scripts/python.exe scripts/chess_hrm_report.py','```\n',
              'Results: `results/chess_hrm_training.json.gz`, `chess_hrm_final_halting.json.gz`, '
              '`chess_hrm_search_plan.json.gz`, `chess_hrm_search_games.json.gz`, '
              '`chess_hrm_search_summary.json.gz`, `chess_hrm_final_audit.json.gz`, `chess_hrm_report.json.gz`. '
              'The model is `results/chess_hrm_adaptive_model.pt`. All per-game PGNs, player configs, move timings and '
              'executed halting histograms are retained in the compressed game archive.']
    path=Path('docs/CHESS_STRENGTH.md');previous=path.read_text(encoding='utf-8')
    marker='## Adaptive HRM in search with learned halting (2026-10-28)'
    if marker in previous:raise ValueError('report section already present')
    path.write_text(previous+'\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':report()
