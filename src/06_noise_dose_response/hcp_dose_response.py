"""Week 6, HCP: dose-response of the model's gain on leakage, from the week-3 test outputs (no new fits).
Coherent vertices that are clean at 1.6 mm (standard-mapping far leak < 2%) are binned by the far leak of standard mapping at the
coarse grid (other hemisphere or > 15 mm geodesic; week 2). Per bin and subject: week-3 cross-run residual of the model (chosen
settings) and of the competitors (upsampling + 1.6 mm mapping + smoothing 8 mm; standard + smoothing 10 mm; standard mapping).
Usage (xcp_d container): python3 hcp_dose_response.py OUTDIR [EST_DIR [SUBJECTS_FILE [STIMULUS]]]
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
from evaluate import Merged, subject_context, cross_run, methods, add_upsmooth, set_stimulus
from w3lib import *

R3 = f'{W_ROOT}/results/week3'
CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    est = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else f'{R3}/test_all'                  # week 13: batches, rings
    subs = ([l.strip() for l in open(sys.argv[3]) if l.strip()] if len(sys.argv) > 3 and sys.argv[3] else
            json.load(open(f'{R3}/_eval/week3_test_scores.json'))['test_subjects'])
    set_stimulus(sys.argv[4] if len(sys.argv) > 4 else 'wedge')
    res = {}
    for S in subs:
        E = Merged([f'{est}/{S}/{S}.week3_estimates.npz'])
        ctx = subject_context(S, E['idx']); add_upsmooth(E, ctx)
        base = (ctx['coh'] > COH_THR) & (ctx['lf16'] < 0.02)
        res[S] = {}
        for g in ('2.4', '3.2'):
            mt = methods(E, g, ctx)
            names = {'model': CHOSEN[g], 'upsample_smoothing': 'upsmooth8', 'smoothing': 'smooth10', 'standard': 'standard'}
            r = {}
            for lo, hi in BINS:
                sel = base & (ctx['lf'][g] >= lo) & (ctx['lf'][g] < hi)
                r[f'{lo:g}-{hi:g}'] = dict(n=int(sel.sum()), **{lab: (cross_run(E, mt[m], sel) if sel.sum() >= 30 else None)
                                                                 for lab, m in names.items()})
            res[S][g] = r
        print(S, {g: {b: (v['n'], None if v['model'] is None else round(v['model'] - min(v['upsample_smoothing'], v['smoothing']), 4))
                      for b, v in res[S][g].items()} for g in res[S]}, flush=True)
    json.dump(res, open(f'{out}/hcp_dose_response.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
