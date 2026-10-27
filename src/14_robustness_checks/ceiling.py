"""Week 14 (reviewer pass): the HCP noise ceiling and the share of the competitor-to-ceiling gap that the model closes.

Ceiling: one 1.6 mm wedge run predicting the other run's 1.6 mm map (same score, same vertex sets; the predicting run raw or
surface-smoothed 2–8 mm), from the week-3 evaluation (`results/week3/week3_ceiling.json`, 15 development subjects). The model and
the competitors predict the same target from a coarsened run (`week3_test_scores.json`). Gap closed = (competitor − model) /
(competitor − ceiling), from the means over subjects; the per-subject median of the same ratio is given as well.
Usage (local): python3 src/14_robustness_checks/ceiling.py OUT_PREFIX
"""
import sys, json
import numpy as np

C = json.load(open('results/week3/week3_ceiling.json'))['per_subject']
T = json.load(open('results/week3/week3_test_scores.json'))['scores']
subs = sorted(T); G = ('2.4', '3.2')
STRATA = [('coherent', 'all coherent vertices'), ('clean16_leak5', 'clean at 1.6 mm, far leak > 5%'), ('clean16_leak10', 'clean at 1.6 mm, far leak > 10%'), ('calc', 'pericalcarine')]
CEIL = [('1.6_raw', 'raw'), ('1.6_smooth4', 'smoothed 4 mm'), ('1.6_smooth6', 'smoothed 6 mm')]
f = lambda x: f'{x:.3f}'
res = {'subjects': subs, 'rows': []}
L = ['# Week 14: HCP noise ceiling and the gap the model closes (15 development subjects)', '',
     'Cross-run residual against the other run\'s 1.6 mm map (mean over subjects). "Ceiling": the other 1.6 mm run itself as the predictor,',
     'raw or surface-smoothed. Competitors and model predict from the coarsened run (Section 4.3). Gap closed = (competitor − model) /',
     '(competitor − ceiling) from the means; in brackets the median over subjects of the per-subject ratio. Source: `src/14_robustness_checks/ceiling.py`.', '',
     '| grid | vertex set | ceiling raw | ceiling 4 mm | ceiling 6 mm | upsampling + smoothing 8 mm | per-subject best competitor | model | gap closed vs 6 mm ceiling (competitor = upsampling + smoothing) | gap closed (per-subject best competitor) |',
     '|---|---|---|---|---|---|---|---|---|---|']
for g in G:
    for s, sl in STRATA:
        ce = {k: np.array([C[S][f'{g}:{s}:{k}'] for S in subs]) for k, _ in CEIL}
        ups = np.array([T[S][g]['upsample_smoothing'][s] for S in subs]); sm = np.array([T[S][g]['smoothing'][s] for S in subs])
        best = np.minimum(ups, sm); mod = np.array([T[S][g]['model'][s] for S in subs])
        c6 = ce['1.6_smooth6']
        gap_u = (ups.mean() - mod.mean()) / (ups.mean() - c6.mean()); gap_b = (best.mean() - mod.mean()) / (best.mean() - c6.mean())
        med_u = float(np.median((ups - mod) / (ups - c6))); med_b = float(np.median((best - mod) / (best - c6)))
        row = dict(grid=g, stratum=s, ceiling={k: float(v.mean()) for k, v in ce.items()}, upsample_smoothing=float(ups.mean()), best_competitor=float(best.mean()),
                   model=float(mod.mean()), gap_closed_upsmooth=float(gap_u), gap_closed_best=float(gap_b), gap_closed_upsmooth_median=med_u, gap_closed_best_median=med_b)
        res['rows'].append(row)
        L.append(f"| {g} mm | {sl} | {f(ce['1.6_raw'].mean())} | {f(ce['1.6_smooth4'].mean())} | {f(c6.mean())} | {f(ups.mean())} | {f(best.mean())} | {f(mod.mean())} | "
                 f"{100 * gap_u:.0f}% [{100 * med_u:.0f}%] | {100 * gap_b:.0f}% [{100 * med_b:.0f}%] |")
open(f'{sys.argv[1]}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{sys.argv[1]}.json', 'w'), indent=1)
print('\n'.join(L))
