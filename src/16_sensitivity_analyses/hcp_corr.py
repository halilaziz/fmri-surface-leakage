"""Week 16, F: the week-3 inversion with the correlated-noise likelihood of week 6 (w6lib.ProblemCorr), one HCP subject.

The noise correlation on each emulated grid is estimated from the data themselves: the complex Fourier components at the eight
frequency bins neighbouring the stimulus frequency (the same bins that give the per-voxel noise power), real and imaginary
parts of both runs, each scaled to unit variance per voxel (32 real volumes), radially averaged in physical frequency and
floored at 0.1 of white (w6lib.noise_spectrum). The marginal variance, prior, settings, tolerance and rendering are those of the
week-3 fit; only the likelihood changes. Writes render_{g}_r{rho}_s{ss}corr_{run} (and f_...), plus the radial noise spectrum.
Usage (xcp_d container): python3 hcp_corr.py SUBJECT OUTDIR --settings '2.4=12:0.125;3.2=24:0.25' [--threads 8]
"""
import os, sys, json, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '03_hcp_emulation')); sys.path.insert(0, os.path.join(HERE, '..', '06_noise_dose_response'))
from w3lib import *
from w6lib import ProblemCorr, noise_spectrum

W2SUB = f'{W_ROOT}/results/week2/subjects'


def cmv(M, x):
    return (M @ np.ascontiguousarray(x.real, np.float32)) + 1j * (M @ np.ascontiguousarray(x.imag, np.float32))


def gifti_maps(p):
    return {d.meta['Name']: np.asarray(d.data, np.float64) for d in nib.load(p).darrays}


def voxel_data_with_bins(S, grids, workers, runs):
    """As w3lib.voxel_wedge_data (full runs) but also keeps the complex values at the noise bins (nvox x 8) per grid and run."""
    off, nwin, cyc, nbins = SEGMENTS['full']
    img = nib.load(f'{HCP}/{S}/MNINonLinear/Results/{CONCAT}/{CONCAT}_hp2000_clean.nii.gz')
    out = {}
    for k, (run, o) in enumerate(runs.items()):
        Y = np.ascontiguousarray(np.asarray(img.dataobj[..., o + BLANK + off:o + BLANK + off + nwin], dtype=np.float32))
        ser = {'1.6': Y}; ser.update(emulate_series(Y, {g: grids[g]['shape'] for g in grids if g != '1.6'}, workers=workers))
        if k == 0:
            m16 = (np.abs(Y).max(3) > 0); out['mask'] = {'1.6': m16.ravel()}
            for g in grids:
                if g != '1.6':
                    out['mask'][g] = emulate(m16.astype(np.float32), grids[g]['shape']).ravel() > 0.9
        for g, X in ser.items():
            X = X.reshape(-1, nwin)
            yv = np.zeros(len(X), np.complex64); nv_ = np.zeros(len(X), np.float32); nb = np.zeros((len(X), len(nbins)), np.complex64)
            for c0 in range(0, len(X), 200000):
                Xi = X[c0:c0 + 200000].astype(np.float64); Fx = np.fft.rfft(Xi - Xi.mean(1)[:, None], axis=1)
                yv[c0:c0 + 200000] = Fx[:, cyc]; nv_[c0:c0 + 200000] = np.mean(np.abs(Fx[:, nbins]) ** 2, axis=1); nb[c0:c0 + 200000] = Fx[:, nbins]
            out[(g, run)] = dict(y=yv, noise=nv_, bins=nb if g != '1.6' else None)
        del Y, ser
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('outdir'); ap.add_argument('--settings', required=True)
    ap.add_argument('--grids', default='2.4,3.2'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S = a.subject
    RUNS, OTHER, W16TAG = stim_runs('wedge'); R0, R1 = list(RUNS)
    coarse = a.grids.split(',')
    parse = lambda txt: [tuple(float(x) for x in s.split(':')) for s in txt.split(',')]
    settings = {kv.split('=')[0]: parse(kv.split('=')[1]) for kv in a.settings.split(';')}
    sd = os.path.join(a.outdir, S); os.makedirs(sd, exist_ok=True)
    T0 = time.time(); log = {'subject': S, 'settings': settings, 'solves': [], 'spectrum': {}}
    say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)
    sbref, grids = subject_grids(S)
    geo = {h: hemi_geometry(S, h) for h in 'LR'}; nv = {h: len(geo[h]['white']) for h in 'LR'}
    scales = subject_scales(S, geo, f'{W_ROOT}/results/week1/maps')
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])]); n = len(idx)
    nroi = {h: int(av[h].sum()) for h in 'LR'}
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(n)
    Cr, Gr = {}, {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h]); Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    P, V, C, Z, _ = subject_points(geo); keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    wk2 = {h: {**gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_wedge.func.gii'), **gifti_maps(f'{W2SUB}/{S}/{S}.{h}.week2_leakage.func.gii')} for h in 'LR'}
    coh = cat({h: wk2[h]['coherence_1.6_all'] for h in 'LR'}); base = coh > COH_THR
    data = voxel_data_with_bins(S, {g: grids[g] for g in ['1.6'] + coarse}, a.threads, RUNS)
    Wmap = {}
    for h in 'LR':
        for tag in sorted(set(W16TAG.values()) | {'all'}):
            Wmap[('1.6', tag, h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_1.6_{h}_{tag}.npz').tocsr()
        for g in coarse:
            Wmap[(g, 'all', h)] = sp.load_npz(f'{W2SUB}/{S}/hcpW_{g}_{h}_all.npz').tocsr()
    mapv = lambda g, tag, vol: cat({h: Wmap[(g, tag, h)] @ vol for h in 'LR'})
    tabs, rad = grid_tabs('1.6', grids['1.6'], scales); K16 = build_operator(P, V, Cc, n, grids['1.6'], tabs, rad).tocsr()
    render = lambda f, r: mapv('1.6', W16TAG[OTHER[r]], cmv(K16, f))
    out = {'idx': idx.astype(np.int32), 'nroi_L': np.int32(nroi['L'])}
    say('setup done; unknowns', n)
    for g in coarse:
        t0 = time.time(); tabs, rad = grid_tabs(g, grids[g], scales)
        K = build_operator(P, V, Cc, n, grids[g], tabs, rad).tocsr()
        noise = 0.5 * (data[(g, R0)]['noise'] + data[(g, R1)]['noise']); valid = data['mask'][g]
        shape = tuple(grids[g]['shape']); vox = voxel_sizes(grids[g])
        # unit residual volumes: noise-bin components of both runs, real and imaginary parts, scaled per voxel
        w = []
        for r in RUNS:
            nb = data[(g, r)]['bins'].astype(np.complex128); sd_ = np.sqrt(np.maximum(data[(g, r)]['noise'], 1e-30) / 2.0)
            ok = valid & (data[(g, r)]['noise'] > 0)
            for b in range(nb.shape[1]):
                for part in (np.real, np.imag):
                    w.append(np.where(ok, part(nb[:, b]) / np.where(ok, sd_, 1.0), 0.0).reshape(shape))
        Pk, kc, prof = noise_spectrum(w, valid.reshape(shape), vox, shape)
        log['spectrum'][g] = dict(k_cycles_per_mm=kc.tolist(), power_rel_white=prof.tolist(), n_volumes=len(w))
        say(g, 'spectrum: power at the lowest / highest bins %.1f / %.2f of white' % (prof[np.isfinite(prof)][0], prof[np.isfinite(prof)][-1]))
        del w
        zg0 = mapv(g, 'all', data[(g, R0)]['y']); dil = cat({h: wk2[h][f'dilution_{g}_all'] for h in 'LR'})
        sig0 = float(np.median(np.abs(zg0[base]) / np.maximum(dil[base], 0.3))); log[f'sigma0_{g}'] = sig0
        prob = ProblemCorr(K, valid, noise, Cr, Gr, nroi, shape, Pk); del K
        B = np.concatenate([prob.rhs(data[(g, r)]['y']) for r in RUNS], 1)
        X = None
        for rho, ss in settings[g]:
            t1 = time.time(); prob.set_prior(rho, ss * sig0); tp = time.time() - t1
            X, it, rel = prob.solve(B, tol=1e-4, maxit=400, X0=X)
            key = f'{g}_r{rho:g}_s{ss:g}corr'
            for i, r in enumerate(RUNS):
                f = X[:, 2 * i] + 1j * X[:, 2 * i + 1]
                out[f'f_{key}_{r}'] = f.astype(np.complex64); out[f'render_{key}_{r}'] = render(f, r).astype(np.complex64)
            log['solves'].append(dict(grid=g, rho=rho, sigma_scale=ss, iterations=it, max_rel_residual=float(rel.max()), rows=int(len(prob.rows)),
                                      t_prior_s=round(tp, 1), t_solve_s=round(time.time() - t1 - tp, 1)))
            say(g, f'corr rho {rho:g} s {ss:g}: {it} iterations, {time.time() - t1:.0f}s')
        del prob
    np.savez_compressed(f'{sd}/{S}.week16_corr.npz', **out)
    log['t_total_s'] = round(time.time() - T0, 1)
    json.dump(log, open(f'{sd}/{S}.week16_corr_log.json', 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main()
