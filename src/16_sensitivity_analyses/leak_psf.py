"""Week 16, B (follow-up): the leakage maps of standard mapping with the PSF model narrowed to the scan's optimum (kernels stretched by
α = 0.85 and 0.7 on all axes) at 1.6 / 2.4 / 3.2 mm, sphere criterion at 15 mm, one subject (operator only).
Usage (xcp_d container): python3 leak_psf.py SUBJECT OUTDIR
"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'; W1MAPS = f'{W_ROOT}/results/week1/maps'
GEO_THR = 15.0; ALPHAS = [1.0, 0.85, 0.7]


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
        grid = grids[g]; log['grids'][g] = {}
        for alpha in ALPHAS:
            tabs, rad = grid_tabs(g, grid, scales)
            if alpha != 1.0:
                tabs = [(t * alpha, v / alpha) for t, v in tabs]; rad = tuple(int(np.ceil(r * alpha)) for r in rad)
            K = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grid, tabs, rad).tocsr()
            out = {'psf_fwhm_mm_xyz': [round(2 * float(tb[0][np.argmax(tb[1] < tb[1][0] / 2)]), 3) for tb in tabs]}
            for h in 'LR':
                Wm = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr(); roi = geo[h]['roi']
                own, leak, leak_abs = np.zeros(nv[h]), np.zeros(nv[h]), np.zeros(nv[h]); idx = np.flatnonzero(roi)
                for c0 in range(0, len(idx), 3000):
                    rows = idx[c0:c0 + 3000]; nr = len(rows)
                    A = (Wm[rows] @ K).tocoo(); j = A.col; val = A.data.astype(np.float64)
                    other = (j >= nv['L']) if h == 'L' else (j < nv['L']); jl = np.where(j >= nv['L'], j - nv['L'], j)
                    far = other | (fscale[h] * np.arccos(np.clip(np.einsum('nd,nd->n', su[h][rows[A.row]], su[h][np.where(other, 0, jl)]), -1, 1)) > GEO_THR)
                    own[rows] = np.bincount(A.row, val * ~far, minlength=nr); leak[rows] = np.bincount(A.row, val * far, minlength=nr)
                    leak_abs[rows] = np.bincount(A.row, np.abs(val * far), minlength=nr)
                tot = own + leak; lf = np.where(np.abs(tot) > 1e-6, leak / np.where(np.abs(tot) > 1e-6, tot, 1.0), 0.0)
                gross = np.where(np.abs(tot) > 1e-6, leak_abs / np.where(np.abs(tot) > 1e-6, np.abs(own) + leak_abs, 1.0), 0.0)
                has = np.diff(Wm.indptr) > 0; w = area[h] * roi * has; wc = area[h] * calc[h] * has
                out[h] = dict(dilution_mean=wmean(tot, w), leak_frac_mean=wmean(lf, w), gross_far_frac_mean=wmean(gross, w),
                              frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, w), frac_leak_frac_gt10=wmean((lf > 0.10) * 1.0, w), calc_leak_frac_mean=wmean(lf, wc))
            log['grids'][g][f'psf{alpha:g}'] = out
            say(g, f'psf{alpha:g}', out['psf_fwhm_mm_xyz'], 'leak %.2f/%.2f%% >5%%: %.1f/%.1f%%' % (100 * out['L']['leak_frac_mean'], 100 * out['R']['leak_frac_mean'], 100 * out['L']['frac_leak_frac_gt5'], 100 * out['R']['frac_leak_frac_gt5']))
            del K
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{outdir}/{S}.week16_leak_psf.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
