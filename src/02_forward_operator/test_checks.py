"""Week 2 operator checks on the test subjects (operators rebuilt exactly as in run_subject.py; needs its hcpW_*.npz):
  (1) brute force: columns of K recomputed by an independent dense loop over voxels; mass conservation of every column
      (sum_v K[v,c] * voxel volume / column volume = 1 for a unit-integral effective PSF);
  (3) emulation consistency: S(K_1.6 f) (k-space crop of the native forward model) vs K_g f (operator built directly with the
      cropped-band kernel and finite support) for g = 2.4, 3.2 mm, for uniform, smooth, rough and single-column fields;
      compared both in volume space and after standard mapping (the quantity the leakage maps depend on).
Usage (xcp_d container): python3 test_checks.py SUBJECT OUTDIR WEEK1_MAPSDIR [--threads N]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w2lib import *


def brute_column(c, P, V, C, grid, tabs, rad):
    """Dense accumulation for column c: loop over its points and over every voxel in the support box."""
    A = grid['affine']; Ainv = np.linalg.inv(A); shape = grid['shape']
    out = {}
    for p, w in zip(P[C == c], V[C == c]):
        q = Ainv[:3, :3] @ p + Ainv[:3, 3]
        b = np.rint(q).astype(int)
        for i in range(b[0] - rad[0], b[0] + rad[0] + 1):
            for j in range(b[1] - rad[1], b[1] + rad[1] + 1):
                for k in range(b[2] - rad[2], b[2] + rad[2] + 1):
                    if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
                        continue
                    x = A[:3, :3] @ np.array([i, j, k]) + A[:3, 3]
                    val = w * np.prod([eval_kernel(tabs[d], np.array([x[d] - p[d]]))[0] for d in range(3)])
                    lin = np.ravel_multi_index((i, j, k), shape)
                    out[lin] = out.get(lin, 0.0) + val
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('w1maps')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S = a.subject; sd = os.path.join(a.outdir, S)
    T0 = time.time(); rng = np.random.default_rng(int(S))
    _, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    scales = subject_scales(S, geo, a.w1maps)
    P, V, C, Z, nv = subject_points(geo)
    ncols = nv['L'] + nv['R']
    colvol = np.bincount(C, weights=V, minlength=ncols)
    roi = np.concatenate([geo['L']['roi'], geo['R']['roi']])
    K = {}
    for g, grid in grids.items():
        t0 = time.time(); tabs, rad = grid_tabs(g, grid, scales)
        K[g] = build_operator(P, V * depth_weight(Z, 'b0'), C, ncols, grid, tabs, rad).tocsc()
        print(S, g, 'K built in', round(time.time() - t0, 1), 's, nnz', K[g].nnz, flush=True)
    log = {'subject': S, 'check1': {}, 'check3': {}}
    # depth-gradient component b1 on the native grid: columns carry ~zero net mass, displaced toward the pial side
    tabs, rad = grid_tabs('1.6', grids['1.6'], scales)
    K1 = build_operator(P, V * depth_weight(Z, 'b1'), C, ncols, grids['1.6'], tabs, rad).tocsc()
    net = np.asarray(K1.sum(0)).ravel()[roi]; gross = np.asarray(abs(K1).sum(0)).ravel()[roi]
    log['b1_native'] = dict(median_net_over_gross=float(np.median(np.abs(net) / np.maximum(gross, 1e-12))), nnz=int(K1.nnz))
    del K1

    # (1) brute force + mass conservation
    test_cols = rng.choice(np.flatnonzero(roi), 40, replace=False)
    for g, grid in grids.items():
        tabs, rad = grid_tabs(g, grid, scales)
        errs = []
        for c in test_cols:
            bf = brute_column(c, P, V, C, grid, tabs, rad)
            col = K[g][:, c].toarray().ravel()
            ref = np.zeros_like(col); ref[list(bf.keys())] = list(bf.values())
            errs.append(np.abs(col - ref).max() / np.abs(ref).max())
        dV = np.prod(voxel_sizes(grid))
        mass = np.asarray(K[g].sum(0)).ravel() * dV / np.maximum(colvol, 1e-12)
        mass = mass[roi & (colvol > 0)]
        log['check1'][g] = dict(n_brute=len(test_cols), max_rel_err=float(np.max(errs)),
                                mass_p01_p50_p99=np.percentile(mass, [1, 50, 99]).round(5).tolist())
        print(S, g, 'check1', log['check1'][g], flush=True)

    # (3) emulation consistency
    mid = np.vstack([geo[h]['mid'] for h in 'LR'])
    fields = {'uniform': roi * 1.0}
    f = np.zeros(ncols)
    for lam in (8.0, 12.0, 20.0, 30.0):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        f += np.cos(2 * np.pi / lam * mid @ u + rng.uniform(0, 2 * np.pi))
    fields['smooth_8-30mm'] = f * roi
    fields['rough_white'] = rng.choice([-1.0, 1.0], ncols) * roi
    singles = rng.choice(np.flatnonzero(roi), 5, replace=False)
    for c in singles:
        e = np.zeros(ncols); e[c] = 1.0; fields[f'single_{c}'] = e
    Wall = {g: {h: sp.load_npz(f'{sd}/hcpW_{g}_{h}_all.npz').tocsr() for h in 'LR'} for g in grids}
    for name, f in fields.items():
        y16 = (K['1.6'] @ f).reshape(grids['1.6']['shape'])
        res = {}
        for g in ('2.4', '3.2'):
            ye = emulate(y16, grids[g]['shape'], workers=a.threads).ravel()
            yk = K[g] @ f
            used = np.zeros(yk.size, bool)
            for h in 'LR':
                used[np.unique(Wall[g][h].indices)] = True
            near = np.abs(yk) > 0
            r = dict(rel_err_all_voxels=float(np.linalg.norm(ye - yk) / np.linalg.norm(ye)),
                     rel_err_operator_support=float(np.linalg.norm((ye - yk)[near]) / np.linalg.norm(ye[near])),
                     energy_outside_support=float(np.linalg.norm(ye[~near]) ** 2 / np.linalg.norm(ye) ** 2),
                     rel_err_mapped_voxels=float(np.linalg.norm((ye - yk)[used]) / max(np.linalg.norm(ye[used]), 1e-12)))
            if not name.startswith('single'):
                me = np.concatenate([Wall[g][h] @ ye for h in 'LR']); mk = np.concatenate([Wall[g][h] @ yk for h in 'LR'])
                ok = np.concatenate([geo[h]['roi'] for h in 'LR'])
                r['rel_err_mapped_vertices'] = float(np.linalg.norm((me - mk)[ok]) / np.linalg.norm(me[ok]))
                r['mapped_mean_emul_vs_op'] = [float(me[ok].mean()), float(mk[ok].mean())]
            res[g] = r
        log['check3'][name] = res
        print(S, 'check3', name, {g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for g, r in res.items()}, flush=True)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week2_checks.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
