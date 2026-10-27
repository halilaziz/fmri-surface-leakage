"""Reference point for the week-3 scores: the other run's own 1.6 mm HCP map as the predictor (raw and surface-smoothed),
scored exactly like the coarse-data methods (cross-run, conjugated, best complex scaling), on the test subjects."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load, strata, resid, OTHER, WEDGE
from w3lib import surface_smooth

X = load(sys.argv[1]); out = {}
for S, (E, ctx) in X.items():
    res = {}
    preds = {'1.6_raw': {r: E[f'z16_{r}'] for r in WEDGE}}
    for fw in (2.0, 4.0, 6.0, 8.0):
        sm = surface_smooth(ctx['Cr'], ctx['Gr'], fw)
        preds[f'1.6_smooth{fw:g}'] = {r: sm(E[f'z16_{r}'].astype(np.complex128)) for r in WEDGE}
    for g in ('2.4', '3.2'):
        for s, sel in strata(ctx, g).items():
            for name, p in preds.items():
                res[f'{g}:{s}:{name}'] = float(np.mean([resid(E[f'z16_{OTHER[r]}'][sel].astype(complex), np.conj(p[r][sel].astype(complex))) for r in WEDGE]))
    out[S] = res
    print(S, {k: round(v, 3) for k, v in res.items() if ':coherent:' in k and k.startswith('2.4')}, flush=True)
keys = sorted(next(iter(out.values())))
summ = {k: (float(np.mean([out[S][k] for S in out])), float(np.std([out[S][k] for S in out]))) for k in keys}
json.dump(dict(per_subject=out, mean_sd=summ), open(sys.argv[2], 'w'), indent=1)
for k in keys:
    print(f'{k:40s} {summ[k][0]:.3f} ± {summ[k][1]:.3f}')
