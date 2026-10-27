"""Week 16, G1: the leakage maps of standard mapping with the far criterion at other thresholds (sphere proxy: 10, 15, 20 mm) and
with 'far = other hemisphere only', for the emulated kernels at 1.6 / 2.4 / 3.2 mm (operator only), one subject.
Usage (xcp_d container): python3 leak_thresholds.py SUBJECT OUTDIR
"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'; W1MAPS = f'{W_ROOT}/results/week1/maps'
THR = [10.0, 15.0, 20.0]


def wmean(x, w):
    return float((x * w).sum() / max(w.sum(), 1e-30))


def main():
    S, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}; nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, W1MAPS)
    area = {h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'}
    N = f'{HCP}/{S}/MNINonLinear/Native'; calc = {}
    for h in 'LR':
        lab = nib.load(f'{N}/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in lab.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        calc[h] = (np.asarray(lab.darrays[0].data) == key) & geo[h]['roi']
    P, V, C, Z, _ = subject_points(geo); ncols = nv['L'] + nv['R']
    su, fscale = {}, {}
    for h in 'LR':
        s = geo[h]['sphere']; Rs = np.linalg.norm(s, axis=1).mean(); su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        fscale[h] = Rs * np.sqrt(area[h][geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    log = {'subject': S, 'grids': {}}
    for g in ('1.6', '2.4', '3.2'):
        grid = grids[g]; tabs, rad = grid_tabs(g, grid, scales)
        K = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grid, tabs, rad).tocsr()
        out = {}
        for h in 'LR':
            Wm = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr(); roi = geo[h]['roi']
            idx = np.flatnonzero(roi); variants = [f'thr{t:g}' for t in THR] + ['oh']
            own = {v: np.zeros(nv[h]) for v in variants}; leak = {v: np.zeros(nv[h]) for v in variants}
            for c0 in range(0, len(idx), 3000):
                rows = idx[c0:c0 + 3000]; nr = len(rows)
                A = (Wm[rows] @ K).tocoo(); j = A.col; val = A.data.astype(np.float64)
                other = (j >= nv['L']) if h == 'L' else (j < nv['L'])
                jl = np.where(j >= nv['L'], j - nv['L'], j)
                d = fscale[h] * np.arccos(np.clip(np.einsum('nd,nd->n', su[h][rows[A.row]], su[h][np.where(other, 0, jl)]), -1, 1))
                for v, far in [(f'thr{t:g}', other | (d > t)) for t in THR] + [('oh', other)]:
                    own[v][rows] = np.bincount(A.row, val * ~far, minlength=nr); leak[v][rows] = np.bincount(A.row, val * far, minlength=nr)
            has = np.diff(Wm.indptr) > 0; w = area[h] * roi * has; wc = area[h] * calc[h] * has
            for v in variants:
                tot = own[v] + leak[v]; lf = np.where(np.abs(tot) > 1e-6, leak[v] / np.where(np.abs(tot) > 1e-6, tot, 1.0), 0.0)
                out[f'{v}_{h}'] = dict(leak_frac_mean=wmean(lf, w), frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, w), frac_leak_frac_gt10=wmean((lf > 0.10) * 1.0, w),
                                       calc_leak_frac_mean=wmean(lf, wc), leak_mean=wmean(leak[v], w))
        log['grids'][g] = out
        say(g, {v: round(100 * 0.5 * (out[f'{v}_L']['leak_frac_mean'] + out[f'{v}_R']['leak_frac_mean']), 2) for v in variants})
        del K
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{outdir}/{S}.week16_leak_thresholds.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
