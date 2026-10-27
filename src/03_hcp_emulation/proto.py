"""Week 3 prototype on one subject: data, FEM, one grid, one prior setting; solver speed, convergence and a first score."""
import os, sys, time, json
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3lib import *

S = sys.argv[1]; g = sys.argv[2]; rho = float(sys.argv[3]); sscale = float(sys.argv[4])
ymax = float(sys.argv[5]) if len(sys.argv) > 5 else None     # restrict unknowns to MNI y < ymax (posterior cortex)
W2 = f'{W_ROOT}/results/week2/subjects/{S}'; W1 = f'{W_ROOT}/results/week1/maps'
T0 = time.time()
sbref, grids = subject_grids(S)
geo = {h: hemi_geometry(S, h) for h in 'LR'}
nv = {h: len(geo[h]['white']) for h in 'LR'}
scales = subject_scales(S, geo, W1)
av_all = analysis_vertices(geo)                     # all modelled cortex (columns of K)
av = analysis_vertices(geo, ymax)                   # unknowns
all_idx = np.concatenate([np.flatnonzero(av_all['L']), nv['L'] + np.flatnonzero(av_all['R'])])
roi_idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])])
nroi = {h: int(av[h].sum()) for h in 'LR'}
loc = -np.ones(nv['L'] + nv['R'], int); loc[all_idx] = np.arange(len(all_idx))
cols = loc[roi_idx]
Cr, Gr = {}, {}
for h in 'LR':
    Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h])
    Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    assert (Cr[h] > 0).all()
print('FEM done', round(time.time() - T0), 's; ROI vertices', nroi, '; area cm2', {h: round(Cr[h].sum() / 100) for h in 'LR'}, flush=True)

t0 = time.time(); data = voxel_wedge_data(S, {k: grids[k] for k in ('1.6', g)}, workers=8)
print('voxel data', round(time.time() - t0), 's; valid voxels', {k: int(v.sum()) for k, v in data['mask'].items()}, flush=True)

P, V, C, Z, _ = subject_points(geo)
t0 = time.time(); tabs, rad = grid_tabs(g, grids[g], scales)
keep = loc[C] >= 0
K = build_operator(P[keep], V[keep], loc[C[keep]], len(all_idx), grids[g], tabs, rad).tocsr()
print('K built', round(time.time() - t0), 's, nnz', K.nnz, flush=True)
t0 = time.time(); prob = Problem(K, data['mask'][g], 0.5 * (data[(g, 'ccw')]['noise'] + data[(g, 'cw')]['noise']), Cr, Gr, nroi,
                                     cols=None if ymax is None else cols); del K
print('problem', round(time.time() - t0), 's; rows', len(prob.rows), 'unknowns', prob.n, 'nnz', prob.K.nnz, flush=True)
t0 = time.time(); _ = prob.KtDK(np.ones((prob.n, 4))); print('one K\'DK product (4 columns)', round(time.time() - t0, 2), 's', flush=True)

# standard mapping and week-2 maps
Wg = {h: sp.load_npz(f'{W2}/hcpW_{g}_{h}_all.npz').tocsr() for h in 'LR'}
W16 = {h: {r: sp.load_npz(f'{W2}/hcpW_1.6_{h}_{"gv" if r == "ccw" else "gvcw"}.npz').tocsr() for r in WEDGE} for h in 'LR'}
m2 = {h: {d.meta['Name']: np.asarray(d.data, float) for d in nib.load(f'{W2}/{S}.{h}.week2_wedge.func.gii').darrays} for h in 'LR'}
l2 = {h: {d.meta['Name']: np.asarray(d.data, float) for d in nib.load(f'{W2}/{S}.{h}.week2_leakage.func.gii').darrays} for h in 'LR'}
cat = lambda d: np.concatenate([d['L'], d['R']])[roi_idx]
zg = {r: cat({h: Wg[h] @ data[(g, r)]['y'] for h in 'LR'}) for r in WEDGE}
z16 = {r: cat({h: W16[h][r] @ data[('1.6', r)]['y'] for h in 'LR'}) for r in WEDGE}
coh = cat({h: m2[h]['coherence_1.6_all'] for h in 'LR'}); dil = cat({h: l2[h][f'dilution_{g}_all'] for h in 'LR'})
lf16 = cat({h: l2[h]['leak_fraction_1.6_gv'] for h in 'LR'}); lfg = cat({h: l2[h][f'leak_fraction_{g}_all'] for h in 'LR'})
base = coh > COH_THR
sigma = sscale * np.median(np.abs(zg['ccw'][base]) / np.maximum(dil[base], 0.3))
print('prior sigma', sigma, flush=True)

t0 = time.time(); prob.set_prior(rho, sigma); print('prior + preconditioner', round(time.time() - t0), 's', flush=True)
B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in WEDGE], 1)
t0 = time.time(); X, it, rel = prob.solve(B, tol=1e-4, maxit=300, verbose=True)
print('solve', round(time.time() - t0), 's,', it, 'iterations, rel', rel, flush=True)
f = {r: X[:, 2 * i] + 1j * X[:, 2 * i + 1] for i, r in enumerate(WEDGE)}

sm = surface_smooth(Cr, Gr, 4.0)
res = {}
for sname, sel in (('coherent', base), ('clean16_leak10', base & (lf16 < 0.02) & (lfg > 0.10)), ('clean16_leak5', base & (lf16 < 0.02) & (lfg > 0.05))):
    out = {'n': int(sel.sum())}
    for mname, est in (('standard', zg), ('standard_dil', {r: zg[r] / np.maximum(dil, 0.3) for r in WEDGE}),
                       ('standard_smooth4', {r: sm(zg[r]) for r in WEDGE}), ('model', f)):
        cross = 0.5 * (resid(z16['cw'][sel], np.conj(est['ccw'][sel])) + resid(z16['ccw'][sel], np.conj(est['cw'][sel])))
        within = 0.5 * (resid(z16['ccw'][sel], est['ccw'][sel]) + resid(z16['cw'][sel], est['cw'][sel]))
        out[mname] = (round(cross, 4), round(within, 4))
    res[sname] = out
    print(sname, out, flush=True)
json.dump(dict(subject=S, grid=g, rho=rho, sigma_scale=sscale, ymax=ymax, iterations=it, t_total_s=round(time.time() - T0), scores=res),
          open(f'{W_ROOT}/work/w3_proto_{S}_{g}_{rho}_{sscale}_{ymax}.json', 'w'), indent=1)
print('total', round(time.time() - T0), 's')
