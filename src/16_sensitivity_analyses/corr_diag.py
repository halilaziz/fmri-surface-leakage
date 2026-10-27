"""Week 16, F diagnostic (descriptive): how the correlated-noise estimate differs from the white-noise estimate at the same prior
settings, per development subject. For each grid and run: the amplitude ratio |corr| / |white| of the rendered 1.6 mm prediction
over the analysis vertices (root mean square), the magnitude of their complex correlation, and the same after removing each
map's mean over the analysis vertices of its hemisphere (the coarsest spatial scale).
Usage (xcp_d container): python3 corr_diag.py OUT_JSON S1 S2 ...   (reads results/week3/test_all and results/week16/corr npz files)
"""
import os
import sys, json, glob, numpy as np
W = os.environ.get('GEO_ROOT', '/path/to/geo_root') + '/results'
out, subs = sys.argv[1], sys.argv[2:]
res = {}
for S in subs:
    a = np.load(f'{W}/week3/test_all/{S}/{S}.week3_estimates.npz'); b = np.load(f'{W}/week16/corr/{S}/{S}.week16_corr.npz')
    nL = int(b['nroi_L']); r = {}
    for g, st in (('2.4', 'r12_s0.125'), ('3.2', 'r24_s0.25')):
        for k in [k for k in b.files if k.startswith(f'render_{g}_{st}corr_')]:
            run = k.split('corr_')[1]; kw = f'render_{g}_{st}_{run}'
            if kw not in a.files:
                continue
            c = b[k].astype(np.complex128); w = a[kw].astype(np.complex128)
            if len(c) != len(w):
                r[f'{g}_{run}'] = 'length mismatch'; continue
            rms = lambda x: float(np.sqrt(np.mean(np.abs(x) ** 2)))
            cc = lambda x, y: float(np.abs(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)))
            dm = lambda x: np.concatenate([x[:nL] - x[:nL].mean(), x[nL:] - x[nL:].mean()])
            r[f'{g}_{run}'] = dict(amp_ratio=rms(c) / rms(w), corr=cc(c, w), amp_ratio_demeaned=rms(dm(c)) / rms(dm(w)), corr_demeaned=cc(dm(c), dm(w)),
                                   mean_share_white=float(1 - (rms(dm(w)) / rms(w)) ** 2), mean_share_corr=float(1 - (rms(dm(c)) / rms(c)) ** 2))
    res[S] = r; print(S, {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(out, 'w'), indent=1)
