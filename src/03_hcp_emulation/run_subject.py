"""Week 3, per subject: model-based estimation of the wedge response field from the emulated 2.4 / 3.2 mm data.
  - voxel-level complex wedge responses and noise power (CCW, CW runs) at 1.6 mm and on the emulated grids;
  - per coarse grid and prior setting (Matern range rho, s.d. scale): posterior mean of the column-mean field on all cortex;
  - comparison products, all in the domain of the reference (the other run's HCP-standard 1.6 mm surface map):
      standard HCP ribbon mapping of the coarse data, the same + surface smoothing (FWHM 2/4/6/8 mm), k-space upsampling of
      the coarse volume mapped with the 1.6 mm weights, and each estimate rendered through the 1.6 mm model (W_1.6 K_1.6 f);
  - optional, for one chosen setting: averaging kernels at sample vertices and a known-truth simulation.
Everything is saved for evaluate.py; nothing is scored here.
Usage (xcp_d container): python3 run_subject.py SUBJECT OUTDIR --settings 6:1,6:0.5 | '2.4=6:1;3.2=12:0.5' [--grids 2.4,3.2] [--akernels 24]
                         [--simulate] [--threads 8]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3lib import *

W1MAPS = f'{W_ROOT}/results/week1/maps'
W2SUB = f'{W_ROOT}/results/week2/subjects'
OTHER = {'ccw': 'cw', 'cw': 'ccw'}
W16TAG = {'ccw': 'gv', 'cw': 'gvcw'}        # HCP standard: each run mapped with its own goodvoxels


def cmv(M, x):
    """Real sparse matrix (float32) times complex vector without a complex copy of the matrix."""
    return (M @ np.ascontiguousarray(x.real, np.float32)) + 1j * (M @ np.ascontiguousarray(x.imag, np.float32))


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subject'); ap.add_argument('outdir')
    ap.add_argument('--settings', required=True,
                    help="'rho:scale,rho:scale' for every grid, or per grid '2.4=rho:scale,...;3.2=rho:scale,...'; the last "
                         "setting of a grid is used for its averaging kernels and simulation")
    ap.add_argument('--grids', default='2.4,3.2'); ap.add_argument('--akernels', type=int, default=0)
    ap.add_argument('--simulate', action='store_true'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--fwhms', default='2,4,6,8,10,12', help='surface-smoothing FWHMs (mm) of the competitor')
    ap.add_argument('--stimulus', default='wedge', choices=sorted(STIMULI), help='wedge (RETCCW/RETCW) or ring (RETEXP/RETCON)')
    ap.add_argument('--segment', default='full', choices=sorted(SEGMENTS), help='part of each source run used (references stay full)')
    ap.add_argument('--psf-stretch', type=float, default=1.0,
                    help='robustness test: the inversion uses coarse-grid kernels stretched by this factor (data, rendering unchanged)')
    ap.add_argument('--psf-stretch-axes', default='xyz', help='week 16: axes the stretch applies to (subset of xyz; keys get the letters when not all)')
    a = ap.parse_args(); S = a.subject
    RUNS, OTHER, W16TAG = stim_runs(a.stimulus); R0, R1 = list(RUNS)
    FWHMS = tuple(float(x) for x in a.fwhms.split(','))
    coarse = a.grids.split(',')
    parse = lambda txt: [tuple(float(x) for x in s.split(':')) for s in txt.split(',')]
    if '=' in a.settings:
        settings = {kv.split('=')[0]: parse(kv.split('=')[1]) for kv in a.settings.split(';')}
    else:
        settings = {g: parse(a.settings) for g in coarse}
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    T0 = time.time(); log = {'subject': S, 'settings': settings, 'grids': coarse, 'stimulus': a.stimulus, 'segment': a.segment, 'solves': []}
    say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)

    # ---- geometry, unknowns, FEM
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
    keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    n = len(idx)
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    split = lambda x: {'L': x[:nroi['L']], 'R': x[nroi['L']:]}
    wk2 = {h: {**gifti_maps(f"{W2SUB}/{S}/{S}.{h}.{STIMULI[a.stimulus]['maps']}.func.gii"), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')}
           for h in 'LR'}
    coh = cat({h: wk2[h]['coherence_1.6_all'] for h in 'LR'})
    base = coh > COH_THR
    say('unknowns', n, nroi)

    # ---- voxel data and standard HCP mapping
    data = voxel_wedge_data(S, {g: grids[g] for g in ['1.6'] + coarse}, workers=a.threads, runs=RUNS, segment=a.segment)
    # references (the other run's HCP-standard 1.6 mm map) always come from the full runs
    data_ref = data if a.segment == 'full' else voxel_wedge_data(S, {'1.6': grids['1.6']}, workers=a.threads, runs=RUNS, segment='full')
    say('voxel data; valid voxels', {g: int(m.sum()) for g, m in data['mask'].items()})
    Wmap = {}
    for h in 'LR':
        for tag in sorted(set(W16TAG.values()) | {'all'}):
            Wmap[('1.6', tag, h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_1.6_{h}_{tag}.npz').tocsr()
        for g in coarse:
            Wmap[(g, 'all', h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr()
    mapv = lambda g, tag, vol: cat({h: Wmap[(g, tag, h)] @ vol for h in 'LR'})
    out = {'idx': idx.astype(np.int32), 'nroi_L': np.int32(nroi['L'])}
    for r in RUNS:
        out[f'z16_{r}'] = mapv('1.6', W16TAG[r], data_ref[('1.6', r)]['y']).astype(np.complex64)      # reference (HCP standard)
        out[f'z16all_{r}'] = mapv('1.6', 'all', data_ref[('1.6', r)]['y']).astype(np.complex64)

    # ---- 1.6 mm operator for rendering estimates into the reference domain
    t0 = time.time(); tabs, rad = grid_tabs('1.6', grids['1.6'], scales)
    K16 = build_operator(P, V, Cc, n, grids['1.6'], tabs, rad).tocsr()
    say('K_1.6 built', round(time.time() - t0), 's, nnz', K16.nnz)
    render = lambda f, r: mapv('1.6', W16TAG[OTHER[r]], cmv(K16, f))   # as the other run's HCP map would show it

    # averaging kernels and the simulation use the last setting listed
    alpha = a.psf_stretch
    axes = [d for d, c in enumerate('xyz') if c in a.psf_stretch_axes]
    sfx = '' if alpha == 1.0 else f'_psf{alpha:g}' + ('' if len(axes) == 3 else ''.join(c for c in 'xyz' if c in a.psf_stretch_axes))
    for g in coarse:
        t0 = time.time(); tabs, rad = grid_tabs(g, grids[g], scales)
        if alpha != 1.0:       # mis-specified PSF: k(t) -> k(t / alpha) / alpha (unit integral kept), support scaled with it
            tabs = [((t * alpha, v / alpha) if d in axes else (t, v)) for d, (t, v) in enumerate(tabs)]
            rad = tuple(int(np.ceil(r * alpha)) if d in axes else r for d, r in enumerate(rad))
        K = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
        noise = 0.5 * (data[(g, R0)]['noise'] + data[(g, R1)]['noise'])
        if a.akernels:
            # sample vertices: half where standard mapping at this grid leaks > 10%, half random coherent; and standard
            # mapping's averaging kernels there (rows of W_g K_g), taken before the full operator is freed
            lf = cat({h: wk2[h][f'leak_fraction_{g}_all'] for h in 'LR'})
            rng = np.random.default_rng(int(S) + int(float(g) * 10))
            hi = np.flatnonzero(base & (lf > 0.10)); nh = min(len(hi), a.akernels // 2)
            samp = np.concatenate([rng.choice(hi, nh, replace=False), rng.choice(np.flatnonzero(base), a.akernels - nh, replace=False)])
            gidx = idx[samp]
            Wrows = sp.vstack([Wmap[(g, 'all', 'L' if v < nv['L'] else 'R')][v if v < nv['L'] else v - nv['L']] for v in gidx]).tocsr()
            out[f'ak_samples_{g}'] = samp.astype(np.int32)
            out[f'ak_std_{g}'] = np.asarray((Wrows @ K).todense(), np.float32)
        prob = Problem(K, data['mask'][g], noise, Cr, Gr, nroi)
        if not a.simulate:
            del K
        say(g, 'problem ready', round(time.time() - t0), 's; rows', len(prob.rows), 'nnz', prob.K.nnz)
        zg = {r: mapv(g, 'all', data[(g, r)]['y']) for r in RUNS}
        dil = cat({h: wk2[h][f'dilution_{g}_all'] for h in 'LR'})
        sig0 = float(np.median(np.abs(zg[R0][base]) / np.maximum(dil[base], 0.3)))
        log[f'sigma0_{g}'] = sig0
        for r in RUNS:
            out[f'z_{g}_{r}'] = zg[r].astype(np.complex64)
            up = upsample(data[(g, r)]['y'].reshape(grids[g]['shape']), grids['1.6']['shape'], workers=a.threads).ravel()
            out[f'up_{g}_{r}'] = mapv('1.6', W16TAG[OTHER[r]], up.astype(np.complex64)).astype(np.complex64)
        for fw in FWHMS:
            sm = surface_smooth(Cr, Gr, fw)
            for r in RUNS:
                out[f'smooth{fw:g}_{g}_{r}'] = sm(zg[r]).astype(np.complex64)
        B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in RUNS], 1)
        X = None
        for rho, ss in settings[g]:
            t1 = time.time(); prob.set_prior(rho, ss * sig0); tp = time.time() - t1
            X, it, rel = prob.solve(B, tol=1e-4, maxit=400, X0=X)
            key = f'{g}_r{rho:g}_s{ss:g}{sfx}'
            for i, r in enumerate(RUNS):
                f = X[:, 2 * i] + 1j * X[:, 2 * i + 1]
                out[f'f_{key}_{r}'] = f.astype(np.complex64)
                out[f'render_{key}_{r}'] = render(f, r).astype(np.complex64)
            log['solves'].append(dict(grid=g, rho=rho, sigma_scale=ss, iterations=it, max_rel_residual=float(rel.max()),
                                      t_prior_s=round(tp, 1), t_solve_s=round(time.time() - t1 - tp, 1)))
            say(g, f'rho {rho:g} s {ss:g}: prior {tp:.0f}s, {it} iterations, {time.time() - t1 - tp:.0f}s')
        if a.akernels:
            E = np.zeros((n, len(samp))); E[samp, np.arange(len(samp))] = 1.0
            t1 = time.time(); U, it, rel = prob.solve(E, tol=1e-4, maxit=400)
            out[f'ak_model_{g}'] = prob.KtDK(U).T.astype(np.float32)     # rows of the resolution matrix H^-1 K'DK
            say(g, 'averaging kernels', len(samp), 'vertices,', it, 'iterations,', round(time.time() - t1), 's')
        if a.simulate:
            # known truth: the 1.6 mm maps of both runs averaged (the other run conjugated and phase-aligned), smoothed 3 mm and
            # undiluted, so the truth carries little single-run noise; data = K_g f_true + noise with the empirical per-voxel
            # noise power; scored in the reference domain against the noise-free rendering W_1.6 K_1.6 f_true
            rng = np.random.default_rng(int(S) + 7)
            dil16 = cat({h: wk2[h]['dilution_1.6_all'] for h in 'LR'})
            sm3 = surface_smooth(Cr, Gr, 3.0)
            for r in RUNS:
                za, zb = out[f'z16all_{r}'].astype(np.complex128), np.conj(out[f'z16all_{OTHER[r]}'].astype(np.complex128))
                c = np.vdot(zb[base], za[base]) / np.vdot(zb[base], zb[base]); c /= abs(c)
                ft = sm3(0.5 * (za + c * zb)) / np.maximum(dil16, 0.3)
                ysyn = cmv(K, ft).astype(np.complex64)
                sdv = np.sqrt(noise / 2.0)
                ysyn = ysyn + (sdv * (rng.normal(size=len(sdv)) + 1j * rng.normal(size=len(sdv)))).astype(np.complex64)
                ysyn[~data['mask'][g]] = 0
                out[f'sim_truth_render_{g}_{r}'] = mapv('1.6', W16TAG[r], cmv(K16, ft)).astype(np.complex64)
                zs = mapv(g, 'all', ysyn); out[f'sim_z_{g}_{r}'] = zs.astype(np.complex64)
                for fw in FWHMS:
                    out[f'sim_smooth{fw:g}_{g}_{r}'] = surface_smooth(Cr, Gr, fw)(zs).astype(np.complex64)
                ups = upsample(ysyn.reshape(grids[g]['shape']), grids['1.6']['shape'], workers=a.threads).ravel()
                out[f'sim_up_{g}_{r}'] = mapv('1.6', W16TAG[r], ups.astype(np.complex64)).astype(np.complex64)
                Bs = prob.rhs(ysyn); Xs, it, rel = prob.solve(Bs, tol=1e-4, maxit=400)
                fs = Xs[:, 0] + 1j * Xs[:, 1]
                out[f'sim_f_{g}_{r}'] = fs.astype(np.complex64)
                out[f'sim_render_{g}_{r}'] = mapv('1.6', W16TAG[r], cmv(K16, fs)).astype(np.complex64)
                out[f'sim_truth_f_{g}_{r}'] = ft.astype(np.complex64)
            say(g, 'simulation done')
            del K
        del prob
    np.savez_compressed(f'{sd}/{S}.week3_estimates.npz', **out)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week3_log.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
