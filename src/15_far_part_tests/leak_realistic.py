"""Week 15, analysis C: the leakage maps at 2.4 and 3.2 mm for realistic native-acquisition PSFs (no emulation window), one subject.

The published 2.4 / 3.2 mm leak figures are properties of the emulated kernel (native 1.6 mm kernel times a Tukey window at the
coarse Nyquist; effective PSF 1.6–1.7 voxels). Here the same operator machinery is run with the kernel a native acquisition at
that voxel size would have: the week-2 native model (Tukey-tapered sinc in plane, T2* decay over the echo train along PE, erf
slice profile, interpolation blur) with h = 2.4 or 3.2 mm — 'native' —, and the same with partial Fourier 7/8 along PE
approximated by the magnitude PSF of a zero-filled reconstruction — 'native_pf78'. The emulated kernel is recomputed as the
reference ('emul'). Standard mapping uses the saved coarse-grid ribbon weights. Per vertex: own, net far leak, gross far weight
and leak fraction (the week-2 definitions, far = other hemisphere or > 15 mm on the area-scaled sphere); per subject: area-weighted
means, the share of cortex above 5% / 10%, pericalcarine values and the kernel FWHM per axis.
Needs: SBRef header, native surfaces, week-1 ACPC surfaces, week-2 ribbon weights; no time series.
Usage (xcp_d container): python3 leak_realistic.py SUBJECT OUTDIR [--grids 2.4,3.2] [--variants emul,native,native_pf78]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'
W1MAPS = f'{W_ROOT}/results/week1/maps'
GEO_THR = 15.0


def wmean(x, w):
    return float((x * w).sum() / max(w.sum(), 1e-30))


def tabs_for(variant, g, grid, scales):
    dv = voxel_sizes(grid); rad = SUPPORT_VOX[g]
    if variant == 'emul':
        return grid_tabs(g, grid, scales)
    acq = dict(ACQ7T, h=float(g))
    if variant == 'native_pf78':
        acq['pf'] = 7.0 / 8.0
    tabs = [kernel_table(d, scales[d], acq=acq, trunc_mm=(rad[d] + 0.5) * dv[d], R_mm=(rad[d] + 1) * dv[d], rolloff=ROLLOFF['native'])
            for d in range(3)]
    return tabs, rad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('outdir')
    ap.add_argument('--grids', default='2.4,3.2'); ap.add_argument('--variants', default='emul,native,native_pf78')
    a = ap.parse_args(); S = a.subject
    os.makedirs(a.outdir, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, W1MAPS)
    area = {h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'}
    N = f'{HCP}/{S}/MNINonLinear/Native'
    calc = {}
    for h in 'LR':
        lab = nib.load(f'{N}/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in lab.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        calc[h] = (np.asarray(lab.darrays[0].data) == key) & geo[h]['roi']
    P, V, C, Z, _ = subject_points(geo); ncols = nv['L'] + nv['R']
    su, fscale = {}, {}
    for h in 'LR':
        s = geo[h]['sphere']; Rs = np.linalg.norm(s, axis=1).mean()
        su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        fscale[h] = Rs * np.sqrt(area[h][geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    Wts = {(g, h): sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr() for g in a.grids.split(',') for h in 'LR'}
    log = {'subject': S, 'grids': {}, 'maps': {}}
    maps = {h: {} for h in 'LR'}
    for g in a.grids.split(','):
        grid = grids[g]; log['grids'][g] = {}
        for variant in a.variants.split(','):
            t0 = time.time(); tabs, rad = tabs_for(variant, g, grid, scales)
            K = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grid, tabs, rad).tocsr()
            out = {'nnz_K': int(K.nnz), 't_build_K_s': round(time.time() - t0, 1),
                   'psf_fwhm_mm_xyz': [round(2 * float(tb[0][np.argmax(tb[1] < tb[1][0] / 2)]), 3) for tb in tabs]}
            for h in 'LR':
                Wm = Wts[(g, h)]; roi = geo[h]['roi']
                own, leak, leak_abs, leak_oh = (np.zeros(nv[h]) for _ in range(4))
                idx = np.flatnonzero(roi)
                for c0 in range(0, len(idx), 3000):
                    rows = idx[c0:c0 + 3000]; nr = len(rows)
                    A = (Wm[rows] @ K).tocoo(); j = A.col; val = A.data.astype(np.float64)
                    other = (j >= nv['L']) if h == 'L' else (j < nv['L'])
                    jl = np.where(j >= nv['L'], j - nv['L'], j)
                    cosang = np.einsum('nd,nd->n', su[h][rows[A.row]], su[h][np.where(other, 0, jl)])
                    far = other | (fscale[h] * np.arccos(np.clip(cosang, -1, 1)) > GEO_THR)
                    vn, vf = val * ~far, val * far
                    own[rows] = np.bincount(A.row, vn, minlength=nr); leak[rows] = np.bincount(A.row, vf, minlength=nr)
                    leak_abs[rows] = np.bincount(A.row, np.abs(vf), minlength=nr); leak_oh[rows] = np.bincount(A.row, val * other, minlength=nr)
                tot = own + leak
                lf = np.where(np.abs(tot) > 1e-6, leak / np.where(np.abs(tot) > 1e-6, tot, 1.0), 0.0)
                gross = np.where(np.abs(tot) > 1e-6, leak_abs / np.where(np.abs(tot) > 1e-6, np.abs(own) + leak_abs, 1.0), 0.0)
                has = np.diff(Wm.indptr) > 0
                w = area[h] * roi * has; wc = area[h] * calc[h] * has
                out[f'{h}'] = dict(dilution_mean=wmean(tot, w), leak_mean=wmean(leak, w), leak_abs_mean=wmean(leak_abs, w), leak_other_hemi_mean=wmean(leak_oh, w),
                                   leak_frac_mean=wmean(lf, w), gross_far_frac_mean=wmean(gross, w),
                                   frac_leak_frac_gt2=wmean((lf > 0.02) * 1.0, w), frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, w), frac_leak_frac_gt10=wmean((lf > 0.10) * 1.0, w),
                                   frac_gross_gt10=wmean((gross > 0.10) * 1.0, w), frac_gross_gt20=wmean((gross > 0.20) * 1.0, w),
                                   calc_leak_frac_mean=wmean(lf, wc), calc_gross_far_frac_mean=wmean(gross, wc), calc_frac_leak_frac_gt5=wmean((lf > 0.05) * 1.0, wc))
                maps[h][f'leak_fraction_{g}_{variant}'] = lf.astype(np.float32); maps[h][f'gross_far_fraction_{g}_{variant}'] = gross.astype(np.float32)
                maps[h][f'dilution_{g}_{variant}'] = tot.astype(np.float32)
            log['grids'][g][variant] = out
            say(g, variant, 'fwhm', out['psf_fwhm_mm_xyz'], 'leak frac L/R %.2f/%.2f%%' % (100 * out['L']['leak_frac_mean'], 100 * out['R']['leak_frac_mean']),
                '>5%%: %.1f/%.1f%%' % (100 * out['L']['frac_leak_frac_gt5'], 100 * out['R']['frac_leak_frac_gt5']))
            del K
    for h in 'LR':
        darrays = [nib.gifti.GiftiDataArray(v, intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32', meta={'Name': k}) for k, v in maps[h].items()]
        nib.save(nib.gifti.GiftiImage(darrays=darrays), f'{a.outdir}/{S}.{h}.week15_leakage.func.gii')
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{a.outdir}/{S}.week15_leak_realistic.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
