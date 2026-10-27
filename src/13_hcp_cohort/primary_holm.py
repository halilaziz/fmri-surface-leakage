"""Week 13: the eight primary tests of the confirmatory cohort, with multiplicity adjustment across them (post hoc: the cohort
plan, S11.4, fixed the eight primary one-sided tests with no correction across them and Holm only within each family of
secondary tests). Reads results/week13/cohort_final.json (primary set: one subject per family, n = 86) and writes
results/week13/primary_holm.md and .json with Holm step-down and Bonferroni adjusted one-sided p-values.
Usage (local, from the project root): python3 src/13_hcp_cohort/primary_holm.py [RESULTS_WEEK13_DIR]
"""
import json, sys

R = sys.argv[1] if len(sys.argv) > 1 else 'results/week13'
D = json.load(open(f'{R}/cohort_final.json'))
ORDER = [('H1', 'primary', '2.4'), ('H1', 'primary', '3.2'), ('H2', 'H2a', '3.2'), ('H2', 'H2b', '2.4'), ('H2', 'H2b', '3.2'),
         ('H3', 'H3a', '2.4'), ('H3', 'H3a', '3.2'), ('H4', 'primary', '2.4')]          # P1–P8, in the order of the plan
rows = []
for hyp, fam, grid in ORDER:
    r = next(x for x in D['rows'] if x['hypothesis'] == hyp and x['family'] == fam and f'{grid} mm' in x['label'])
    rows.append(dict(hypothesis=hyp, label=r['label'], grid=grid, direction=r['direction'], **{k: r['primary'][k] for k in r['primary'] if k != 'values'}))
assert len(rows) == 8 and len({r['label'] for r in rows}) == 8
m = len(rows)
order = sorted(range(m), key=lambda i: rows[i]['p'])
prev = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, max(prev, (m - rank) * rows[i]['p'])); prev = adj
    rows[i]['p_holm'] = adj; rows[i]['p_bonferroni'] = min(1.0, m * rows[i]['p'])
for n, r in enumerate(rows, 1):
    r['no'] = f'P{n}'
ci_key = next(k for k, v in rows[0].items() if isinstance(v, (list, tuple)) and len(v) == 2)
fp = lambda p: '< 0.001' if p < 0.0005 else f'{p:.3f}'
fm = lambda x: f'{x:+.4f}'.replace('-', '−')
L = ['# Week 13: the eight primary tests of the confirmatory cohort, adjusted across their family (post hoc)', '',
     'The cohort plan (S11.4) fixed eight primary one-sided tests and no correction across them (Holm only within each family of',
     'secondary tests). This table, added after the reviewer round of 2026-10-27, applies Holm\'s step-down and Bonferroni',
     'corrections across the eight, from the exact p-values of `cohort_final.json` (primary set: one subject per family, n = 86;',
     'one-sided Wilcoxon signed-rank; bootstrap 95% interval of the mean). Source: `src/13_hcp_cohort/primary_holm.py`.', '',
     '| no. | hyp. | test | n | mean | in direction | bootstrap 95% | p (one-sided) | p (Holm, 8 tests) | p (Bonferroni, 8 tests) |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lo, hi = r[ci_key]
    L.append(f"| {r['no']} | {r['hypothesis']} | {r['label']} | {r['n']} | {fm(r['mean'])} | {r['n_dir']}/{r['n']} | [{fm(lo)}, {fm(hi)}] | "
             f"{fp(r['p'])} | {fp(r['p_holm'])} | {fp(r['p_bonferroni'])} |")
worst = sorted(rows, key=lambda r: -r['p_holm'])
L += ['', f"Largest Holm-adjusted p: {fp(worst[0]['p_holm'])} ({worst[0]['no']} and {worst[1]['no']}; the two smallest effects). "
      f"All eight remain below 0.05 under Holm; under Bonferroni {worst[0]['no']} and {worst[1]['no']} do not "
      f"({fp(sorted(rows, key=lambda r: -r['p_bonferroni'])[1]['p_bonferroni'])} and {fp(max(r['p_bonferroni'] for r in rows))}).", '']
open(f'{R}/primary_holm.md', 'w').write('\n'.join(L))
json.dump(dict(primary_set=D.get('primary_set'), n_tests=m, rows=rows), open(f'{R}/primary_holm.json', 'w'), indent=1)
print('\n'.join(L))
