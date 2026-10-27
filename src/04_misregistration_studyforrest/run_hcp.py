"""Week 4 on the HCP 7T retinotopy test subjects (emulated grids; week-3 chosen prior settings), one mode per call:
  uncertainty  posterior and noise-only s.d. per vertex (16 perturbation-optimisation samples) at the chosen setting, and the
               exact noise s.d. of standard ribbon mapping; for calibration against run-to-run differences and the week-3 simulation
  shift        misregistration: the inversion uses surfaces rigidly shifted by 0.5 and 1.0 mm (random direction per subject);
               data, rendering and scoring stay nominal
  shiftcomp    the competitors under the same misregistration: an inversion with geometry P + d on data y is the nominal one on
               y(x + d), so the coarse data are resampled at x + d (band-limited Fourier shift) and mapped as usual (standard and
               upsampling + 1.6 mm mapping; smoothing is applied in evaluate_hcp.py)
  depth        a second unknown field for the white-to-pial gradient (b1, orthogonalised to the column mean under each column's
               volume measure) with its own Matern prior (s.d. = gamma x that of b0); estimates of both components, rendered
Usage (xcp_d container): python3 run_hcp.py SUBJECT OUTDIR MODE [--grids 2.4,3.2] [--threads 8]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, scipy.fft as sfft, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
from w3lib import *

W1MAPS = f'{W_ROOT}/results/week1/maps'
W2SUB = f'{W_ROOT}/results/week2/subjects'
CHOSEN = {'2.4': (12.0, 0.125), '3.2': (24.0, 0.25)}     # week-3 tuning, overall criterion
OTHER = {'ccw': 'cw', 'cw': 'ccw'}
W16TAG = {'ccw': 'gv', 'cw': 'gvcw'}
NSAMP = 16
GAMMA = 0.5                                             # prior s.d. of b1 relative to b0


def cmv(M, x):
    return (M @ np.ascontiguousarray(x.real, np.float32)) + 1j * (M @ np.ascontiguousarray(x.imag, np.float32))


def fourier_shift(vol, dv, workers=1):
    """vol sampled at voxel coordinates i + dv (band-limited interpolation; the emulated data are band-limited by construction)."""
    X = sfft.fftn(vol, axes=(0, 1, 2), workers=workers)
    for ax in range(3):
        sh = [1, 1, 1]; sh[ax] = vol.shape[ax]
        X = X * np.exp(2j * np.pi * sfft.fftfreq(vol.shape[ax]) * dv[ax]).reshape(sh)
    return sfft.ifftn(X, axes=(0, 1, 2), workers=workers)


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('mode', choices=['uncertainty', 'shift', 'shiftcomp', 'depth'])
    ap.add_argument('--grids', default='2.4,3.2'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S = a.subject; coarse = a.grids.split(',')
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    T0 = time.time(); log = {'subject': S, 'mode': a.mode, 'grids': coarse, 'steps': []}
    say = lambda *x: print(S, a.mode, f'{time.time() - T0:7.0f}s', *x, flush=True)

    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, W1MAPS)
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])])
    nroi = {h: int(av[h].sum()) for h in 'LR'}
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(len(idx))
    Cr, Gr = {}, {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h]); Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    P, V, C, Z, _ = subject_points(geo)
    keep = loc[C] >= 0; P, V, Cc, Z = P[keep], V[keep], loc[C[keep]], Z[keep]
    n = len(idx)
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    wk2 = {h: {**gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_wedge.func.gii'), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')} for h in 'LR'}
    base = cat({h: wk2[h]['coherence_1.6_all'] for h in 'LR'}) > COH_THR

    data = voxel_wedge_data(S, {g: grids[g] for g in ['1.6'] + coarse}, workers=a.threads)
    Wmap = {}
    for h in 'LR':
        for tag in ('gv', 'gvcw'):
            Wmap[('1.6', tag, h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_1.6_{h}_{tag}.npz').tocsr()
        for g in coarse:
            Wmap[(g, 'all', h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr()
    mapv = lambda g, tag, vol: cat({h: Wmap[(g, tag, h)] @ vol for h in 'LR'})
    out = {'idx': idx.astype(np.int32)}
    if a.mode in ('shift', 'depth'):
        t0 = time.time(); tabs16, rad16 = grid_tabs('1.6', grids['1.6'], scales)
        K16 = build_operator(P, V, Cc, n, grids['1.6'], tabs16, rad16).tocsr()
        say('K_1.6 built', round(time.time() - t0), 's')
        render = lambda f, r, KK=K16: mapv('1.6', W16TAG[OTHER[r]], cmv(KK, f))
    if a.mode == 'depth':
        # b1 orthogonal to b0 within each column: subtract the column's volume-weighted mean of sqrt(3)(2 zeta - 1)
        b1 = depth_weight(Z, 'b1')
        mean_b1 = np.bincount(Cc, weights=V * b1, minlength=n) / np.maximum(np.bincount(Cc, weights=V, minlength=n), 1e-12)
        b1o = b1 - mean_b1[Cc]
        K16b1 = build_operator(P, V * b1o, Cc, n, grids['1.6'], tabs16, rad16).tocsr()
        log['b1_orthogonality'] = float(np.abs(np.bincount(Cc, weights=V * b1o, minlength=n)).max())

    for g in coarse:
        rho, ss = CHOSEN[g]
        tabs, rad = grid_tabs(g, grids[g], scales)
        noise = 0.5 * (data[(g, 'ccw')]['noise'] + data[(g, 'cw')]['noise'])
        zg = {r: mapv(g, 'all', data[(g, r)]['y']) for r in WEDGE}
        dil = cat({h: wk2[h][f'dilution_{g}_all'] for h in 'LR'})
        sig0 = float(np.median(np.abs(zg['ccw'][base]) / np.maximum(dil[base], 0.3)))
        log[f'sigma0_{g}'] = sig0
        t0 = time.time()
        if a.mode == 'uncertainty':
            K = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
            prob = Problem(K, data['mask'][g], noise, Cr, Gr, nroi); del K
            prob.set_prior(rho, ss * sig0)
            B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in WEDGE], 1)
            X, it, _ = prob.solve(B, tol=1e-4, maxit=400)
            for i, r in enumerate(WEDGE):
                out[f'f_{g}_{r}'] = (X[:, 2 * i] + 1j * X[:, 2 * i + 1]).astype(np.complex64)
                out[f'z_{g}_{r}'] = zg[r].astype(np.complex64)
            psd, nsd, it2 = prob.uncertainty(NSAMP, np.random.default_rng(int(S) + int(float(g) * 10)))
            out[f'post_sd_{g}'] = psd.astype(np.float32); out[f'noise_sd_{g}'] = nsd.astype(np.float32)
            # standard mapping: exact noise s.d. of W y for independent voxel noise with the same per-voxel power
            s2 = np.where(data['mask'][g], noise, 0.0)
            out[f'std_noise_sd_{g}'] = np.sqrt(cat({h: (Wmap[(g, 'all', h)].multiply(Wmap[(g, 'all', h)])) @ s2 for h in 'LR'})).astype(np.float32)
            log['steps'].append(dict(grid=g, t_s=round(time.time() - t0), iterations_mean=it, iterations_samples=it2))
            say(g, 'uncertainty done', round(time.time() - t0), 's')
            del prob
        elif a.mode == 'shiftcomp':
            u = np.random.default_rng(int(S)).normal(size=3); u /= np.linalg.norm(u)          # the same direction as 'shift'
            log['shift_direction'] = u.round(4).tolist()
            for dmm in (0.5, 1.0):
                dv = np.linalg.solve(grids[g]['affine'][:3, :3], dmm * u)                       # mm -> coarse voxel units
                for r in WEDGE:
                    d = fourier_shift(data[(g, r)]['y'].reshape(grids[g]['shape']), dv, workers=a.threads)
                    out[f'zshift{dmm:g}_{g}_{r}'] = mapv(g, 'all', d.ravel()).astype(np.complex64)
                    up = upsample(d, grids['1.6']['shape'], workers=a.threads).ravel()
                    out[f'upshift{dmm:g}_{g}_{r}'] = mapv('1.6', W16TAG[OTHER[r]], up.astype(np.complex64)).astype(np.complex64)
            # sanity: a zero shift must reproduce the nominal data
            d0 = fourier_shift(data[(g, 'ccw')]['y'].reshape(grids[g]['shape']), np.zeros(3), workers=a.threads).ravel()
            log[f'zero_shift_rel_err_{g}'] = float(np.linalg.norm(d0 - data[(g, 'ccw')]['y']) / np.linalg.norm(data[(g, 'ccw')]['y']))
            log['steps'].append(dict(grid=g, t_s=round(time.time() - t0)))
            say(g, 'competitor shifts done', round(time.time() - t0), 's')
        elif a.mode == 'shift':
            u = np.random.default_rng(int(S)).normal(size=3); u /= np.linalg.norm(u)
            log['shift_direction'] = u.round(4).tolist()
            for dmm in (0.5, 1.0):
                K = build_operator(P + dmm * u, V, Cc, n, grids[g], tabs, rad).tocsr()
                prob = Problem(K, data['mask'][g], noise, Cr, Gr, nroi); del K
                prob.set_prior(rho, ss * sig0)
                B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in WEDGE], 1)
                X, it, _ = prob.solve(B, tol=1e-4, maxit=400)
                for i, r in enumerate(WEDGE):
                    f = X[:, 2 * i] + 1j * X[:, 2 * i + 1]
                    out[f'render_{g}_shift{dmm:g}_{r}'] = render(f, r).astype(np.complex64)
                log['steps'].append(dict(grid=g, shift_mm=dmm, iterations=it, t_s=round(time.time() - t0)))
                say(g, f'shift {dmm} mm done', it, 'iterations')
                del prob
        else:   # depth
            K0 = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
            K1 = build_operator(P, V * b1o, Cc, n, grids[g], tabs, rad).tocsr()
            K = sp.hstack([K0, K1], format='csr'); del K0, K1
            prob = Problem(K, data['mask'][g], noise, Cr, Gr, nroi); del K
            prob.set_prior_blocks([(Cr['L'], Gr['L'], rho, ss * sig0), (Cr['R'], Gr['R'], rho, ss * sig0),
                                   (Cr['L'], Gr['L'], rho, GAMMA * ss * sig0), (Cr['R'], Gr['R'], rho, GAMMA * ss * sig0)])
            B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in WEDGE], 1)
            X, it, _ = prob.solve(B, tol=1e-4, maxit=600)
            for i, r in enumerate(WEDGE):
                f = X[:, 2 * i] + 1j * X[:, 2 * i + 1]
                f0, f1 = f[:n], f[n:]
                out[f'f0_{g}_{r}'] = f0.astype(np.complex64); out[f'f1_{g}_{r}'] = f1.astype(np.complex64)
                out[f'render_{g}_depth_b0only_{r}'] = render(f0, r).astype(np.complex64)
                out[f'render_{g}_depth_full_{r}'] = (render(f0, r) + mapv('1.6', W16TAG[OTHER[r]], cmv(K16b1, f1))).astype(np.complex64)
            ratio = [float(np.real(np.vdot(out[f'f0_{g}_{r}'][base], out[f'f1_{g}_{r}'][base])) / np.vdot(out[f'f0_{g}_{r}'][base], out[f'f0_{g}_{r}'][base]).real)
                     for r in WEDGE]
            log['steps'].append(dict(grid=g, iterations=it, t_s=round(time.time() - t0), b1_over_b0_coherent=ratio))
            say(g, 'depth done', it, 'iterations; b1/b0 (coherent) =', np.round(ratio, 3))
            del prob
    np.savez_compressed(f'{sd}/{S}.week4_{a.mode}.npz', **out)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week4_{a.mode}_log.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
