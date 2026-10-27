"""Week 7, HCP: the leak-gated estimator on the week-3 test outputs (no new fits).
Per vertex: the model's estimate where the forward operator predicts that standard mapping at the coarse grid leaks more than tau
(far leak: other hemisphere or > 15 mm geodesic, week 2), the best competitor elsewhere (upsampling + 1.6 mm mapping + smoothing,
8 mm, week-3 choice). The model's map is first scaled onto the competitor's (least squares over all vertices; uses no target data),
so that one global scale is fitted in scoring as for every other method. The gate uses only the operator, never the data or target.
Scores: week-3 cross-run residual (other wedge run's 1.6 mm map), all coherent vertices and the week-3 strata, for tau on a grid.
Usage (xcp_d container): python3 hcp_gate.py OUTDIR
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
from evaluate import Merged, subject_context, cross_run, methods, add_upsmooth, strata
from w3lib import *

R3 = f'{W_ROOT}/results/week3'
CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]      # 0 = model everywhere, 1.01 = competitor everywhere


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    subs = json.load(open(f'{R3}/_eval/week3_test_scores.json'))['test_subjects']
    res = {}
    for S in subs:
        E = Merged([f'{R3}/test_all/{S}/{S}.week3_estimates.npz'])
        ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
        res[S] = {}
        for g in ('2.4', '3.2'):
            mt = methods(E, g, ctx); st = strata(ctx, g)
            M, Cm = mt[CHOSEN[g]], mt['upsmooth8']
            gated = {}
            for tau in TAUS:
                gate = ctx['lf'][g] > tau
                pr = {}
                for r in WEDGE:
                    m, c = M[r].astype(complex), Cm[r].astype(complex)
                    k = np.vdot(m, c) / max(np.vdot(m, m).real, 1e-30)            # model scaled onto the competitor (no target)
                    pr[r] = np.where(gate, k * m, c)
                gated[f'{tau:g}'] = {s: (cross_run(E, pr, sel) if sel.sum() >= 30 else None) for s, sel in st.items()}
            res[S][g] = dict(gated=gated, model={s: cross_run(E, M, sel) for s, sel in st.items()},
                             competitor={s: cross_run(E, Cm, sel) for s, sel in st.items()},
                             frac_gated={f'{tau:g}': float(np.mean(ctx['lf'][g][st['coherent']] > tau)) for tau in TAUS})
        print(S, {g: {t: round(v['coherent'], 4) for t, v in res[S][g]['gated'].items()} for g in res[S]}, flush=True)
    json.dump(res, open(f'{out}/hcp_gate.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
