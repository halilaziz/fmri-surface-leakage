"""Week 16: score every model variant present in a set of estimate files of one subject, next to the competitors, in the strata of
the week-3 evaluation and the far-leak bins of the dose-response. Used for the PSF-scale scan (B; tuning subjects, week-3 tuning
files + scan files) and the correlated-noise fits (F; development subjects, week-3 test file + corr file).
Usage (xcp_d container): python3 variant_eval.py S OUT_DIR NPZ1[,NPZ2,...]        -> OUT_DIR/S.variant_eval.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))

BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]

S, out, files = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
from evaluate import Merged, subject_context, methods, add_upsmooth, cross_run, strata
from w3lib import COH_THR
E = Merged(files)
ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
coh = ctx['coh'] > COH_THR; base = coh & (ctx['lf16'] < 0.02)
res = {'subject': S, 'files': files, 'grids': {}}
for g in ('2.4', '3.2'):
    if f'z_{g}_ccw' not in E.files:
        continue
    mt = methods(E, g, ctx)
    st = dict(strata(ctx, g))
    for lo, hi in BINS:
        st[f'bin_{lo:g}-{hi:g}'] = base & (ctx['lf'][g] >= lo) & (ctx['lf'][g] < hi)
    res['grids'][g] = {'n': {k: int(v.sum()) for k, v in st.items()}}
    for m, pred in mt.items():
        res['grids'][g][m] = {k: (cross_run(E, pred, sel) if sel.sum() >= 30 else None) for k, sel in st.items()}
    print(S, g, len(mt), 'methods;', {m: round(res['grids'][g][m]['coherent'], 4) for m in mt if m.startswith('model') and 'psf' in m or m.endswith('corr')}, flush=True)
os.makedirs(out, exist_ok=True)
json.dump(res, open(f'{out}/{S}.variant_eval.json', 'w'), indent=1)
