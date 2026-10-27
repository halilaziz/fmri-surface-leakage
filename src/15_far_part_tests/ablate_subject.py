"""Week 15, analysis A: the week-3 inversion with the far part of the forward operator removed (one subject).

For each coarse grid the full operator K is built as in week 3; then
  - K_near: for every voxel, the entries to unknowns far from the voxel's reference unknown (its largest |K| entry) are set to
    zero; far = other hemisphere or > GEO_THR mm along the sheet on the area-scaled sphere (the leakage-map definition);
  - K_nearn: K_near with every voxel row rescaled to the signed row sum of K (far cortex assumed to respond like near cortex).
Each variant is solved at the frozen settings with the same noise weights, prior and tolerance, and rendered through the full
1.6 mm operator, so that evaluate.py scores it exactly like the week-3 estimates: keys render_{g}_r{rho}_s{ss}near_{run} and
..._nearn_... (plus f_...). Only these arrays are written; the references and competitors are in the week-3 estimate file.
Usage (xcp_d container): python3 ablate_subject.py SUBJECT OUTDIR --settings '2.4=12:0.125;3.2=24:0.25' [--grids 2.4,3.2] [--threads 8]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
from w3lib import *

W2SUB = f'{W_ROOT}/results/week2/subjects'


def cmv(M, x):
    return (M @ np.ascontiguousarray(x.real, np.float32)) + 1j * (M @ np.ascontiguousarray(x.imag, np.float32))


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def near_only(K, sph, hemi, fsc, chunk=40000):
    """K without its far entries (per voxel row, relative to the row's reference unknown). Returns (K_near, stats)."""
    K = K.tocsr(); Kn = K.copy(); Kn.data = K.data.copy()
    absK = abs(K)
    full_mass = np.asarray(absK.sum(1)).ravel()
    far_mass = np.zeros(K.shape[0])
    for r0 in range(0, K.shape[0], chunk):
        r1 = min(r0 + chunk, K.shape[0]); sub = absK[r0:r1]
        if sub.nnz == 0:
            continue
        am = np.asarray(sub.argmax(axis=1)).ravel()                    # reference unknown per row (largest |K|)
        rr = np.repeat(np.arange(r1 - r0), np.diff(sub.indptr)); u = sub.indices; us = am[rr]
        cosang = np.einsum('ij,ij->i', sph[u], sph[us])
        far = (hemi[u] != hemi[us]) | (fsc[hemi[us]] * np.arccos(np.clip(cosang, -1.0, 1.0)) > GEO_THR)
        seg = Kn.data[K.indptr[r0]:K.indptr[r1]]
        far_mass[r0:r1] = np.bincount(rr, np.abs(seg) * far, minlength=r1 - r0)
        seg[far] = 0.0
    Kn.eliminate_zeros()
    return Kn, full_mass, far_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('--settings', required=True)
    ap.add_argument('--grids', default='2.4,3.2'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S = a.subject
    RUNS, OTHER, W16TAG = stim_runs('wedge'); R0, R1 = list(RUNS)
    coarse = a.grids.split(',')
    parse = lambda txt: [tuple(float(x) for x in s.split(':')) for s in txt.split(',')]
    settings = {kv.split('=')[0]: parse(kv.split('=')[1]) for kv in a.settings.split(';')}
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    T0 = time.time(); log = {'subject': S, 'settings': settings, 'grids': coarse, 'variants': ['near', 'nearn'], 'solves': [], 'mass': {}}
    say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)

    # ---- geometry, unknowns, FEM (as week 3)
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, f'{W_ROOT}/results/week1/maps')
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])])
    nroi = {h: int(av[h].sum()) for h in 'LR'}
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(len(idx))
    Cr, Gr = {}, {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h]); Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    P, V, C, Z, _ = subject_points(geo)
    keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    n = len(idx)
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    wk2 = {h: {**gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_wedge.func.gii'), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')} for h in 'LR'}
    coh = cat({h: wk2[h]['coherence_1.6_all'] for h in 'LR'}); base = coh > COH_THR
    # sphere geometry of the unknowns for the far criterion (as in the leakage maps and the kernel analysis)
    su, fs = {}, {}
    for h in 'LR':
        s_ = geo[h]['sphere']; Rs = np.linalg.norm(s_, axis=1).mean(); su[h] = s_ / np.linalg.norm(s_, axis=1, keepdims=True)
        am_, _ = vertex_areas_normals(geo[h]['mid'], geo[h]['faces']); fs[h] = Rs * np.sqrt(am_[geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    sph = np.vstack([su['L'], su['R']])[idx]; hemi = np.where(idx < nv['L'], 0, 1); fsc = np.array([fs['L'], fs['R']])
    say('unknowns', n, nroi)

    # ---- voxel data, mapping weights, 1.6 mm rendering operator
    data = voxel_wedge_data(S, {g: grids[g] for g in ['1.6'] + coarse}, workers=a.threads, runs=RUNS)
    Wmap = {}
    for h in 'LR':
        for tag in sorted(set(W16TAG.values()) | {'all'}):
            Wmap[('1.6', tag, h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_1.6_{h}_{tag}.npz').tocsr()
        for g in coarse:
            Wmap[(g, 'all', h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr()
    mapv = lambda g, tag, vol: cat({h: Wmap[(g, tag, h)] @ vol for h in 'LR'})
    t0 = time.time(); tabs, rad = grid_tabs('1.6', grids['1.6'], scales)
    K16 = build_operator(P, V, Cc, n, grids['1.6'], tabs, rad).tocsr()
    say('K_1.6 built', round(time.time() - t0), 's')
    render = lambda f, r: mapv('1.6', W16TAG[OTHER[r]], cmv(K16, f))
    out = {'idx': idx.astype(np.int32), 'nroi_L': np.int32(nroi['L'])}

    for g in coarse:
        t0 = time.time(); tabs, rad = grid_tabs(g, grids[g], scales)
        K = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
        noise = 0.5 * (data[(g, R0)]['noise'] + data[(g, R1)]['noise'])
        valid = data['mask'][g]
        Kn, full_mass, far_mass = near_only(K, sph, hemi, fsc)
        rs_full = np.asarray(K.sum(1)).ravel(); rs_near = np.asarray(Kn.sum(1)).ravel()
        scale = np.where(rs_near > 1e-3 * np.maximum(rs_full, 1e-30), rs_full / np.where(rs_near > 0, rs_near, 1.0), 1.0)
        Knn = (sp.diags(scale.astype(np.float32)) @ Kn).tocsr()
        v = valid & (full_mass > 0)
        log['mass'][g] = dict(voxels=int(v.sum()), far_share_mean=float((far_mass[v] / full_mass[v]).mean()),
                              voxels_with_far=float((far_mass[v] > 0).mean()), voxels_far_gt10=float((far_mass[v] / full_mass[v] > 0.10).mean()),
                              nnz_full=int(K.nnz), nnz_near=int(Kn.nnz))
        say(g, 'operators built', round(time.time() - t0), 's;', log['mass'][g])
        del K
        zg0 = mapv(g, 'all', data[(g, R0)]['y']); dil = cat({h: wk2[h][f'dilution_{g}_all'] for h in 'LR'})
        sig0 = float(np.median(np.abs(zg0[base]) / np.maximum(dil[base], 0.3))); log[f'sigma0_{g}'] = sig0
        for vname, Kv in (('near', Kn), ('nearn', Knn)):
            prob = Problem(Kv, valid, noise, Cr, Gr, nroi)
            B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in RUNS], 1)
            X = None
            for rho, ss in settings[g]:
                t1 = time.time(); prob.set_prior(rho, ss * sig0); tp = time.time() - t1
                X, it, rel = prob.solve(B, tol=1e-4, maxit=400, X0=X)
                key = f'{g}_r{rho:g}_s{ss:g}{vname}'
                for i, r in enumerate(RUNS):
                    f = X[:, 2 * i] + 1j * X[:, 2 * i + 1]
                    out[f'f_{key}_{r}'] = f.astype(np.complex64); out[f'render_{key}_{r}'] = render(f, r).astype(np.complex64)
                log['solves'].append(dict(grid=g, variant=vname, rho=rho, sigma_scale=ss, iterations=it, max_rel_residual=float(rel.max()),
                                          rows=int(len(prob.rows)), t_prior_s=round(tp, 1), t_solve_s=round(time.time() - t1 - tp, 1)))
                say(g, vname, f'rho {rho:g} s {ss:g}: {it} iterations, {time.time() - t1:.0f}s, rows {len(prob.rows)}')
            del prob
        del Kn, Knn
    np.savez_compressed(f'{sd}/{S}.week15_ablation.npz', **out)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week15_ablation_log.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
