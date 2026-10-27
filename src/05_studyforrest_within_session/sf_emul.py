"""Week 5, Part A: within-session emulation on StudyForrest (one subject, one native 7T session), the HCP week-3 design on real
StudyForrest data with no between-session differences.

Runs are split into halves A (odd) and B (even). The source half is coarsened in k-space by x1.5 and x2 (week 2's crop +
Tukey window, applied per run to betas in raw units, then back to percent change). It then predicts the other half's
standard map at the native resolution. Both directions are done.
Noise of the coarse source: the native run-to-run variance within the source half, propagated exactly through the emulation
(independent voxels), times one calibration factor = observed / propagated run-to-run variance of the coarse runs (this factor
absorbs spatially correlated noise).
Methods, every setting scored (selection happens in summarize_w5.py):
  standard        W_c y_c                               smoothing FWHM grid on it
  upsample        W_n (k-space zero-padded y_c)          smoothing FWHM grid on it
  model           posterior mean from y_c with the emulated PSF, prior grid (range x s.d. scale), rendered W_n K_n f
Source-only split (for self-tuning, no target data): the source half's runs are split again (3 + 2 runs). Every setting predicts one
sub-half's coarse standard map from the other (model rendered W_c K_c f at the source grid; prior s.d. scaled for the sub-half's
noise), averaged over both directions.
Reference: the source half's own native standard map (and its smoothed versions) predicting the target half.
Usage (xcp_d container): python3 sf_emul.py SUBJECT SESSION [--threads 8]      e.g. sub-18 7T-r14
"""
import os, sys, json, time, argparse
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w5lib import *

OUT = f'{ROOT}/results/week5/sf_emul'
FACTORS = (1.5, 2.0)
SAVE_MODEL = [(12.0, 0.125), (24.0, 0.25), (6.0, 1.0), (12.0, 0.5)]      # model predictions kept in the npz (for figures)
GRID_A = PRIOR_GRID + [(3.0, 4.0), (6.0, 4.0)]                              # week-4 grid + two weaker priors (optimum sat at its edge)
XH_COARSE, XH_NATIVE = 0.10, 0.02        # 'mixed across hemispheres': > 10% of the coarse map from the other hemisphere, < 2% at native


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('session'); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(); S, key = a.subject, a.session
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, key, f'{time.time() - T0:7.0f}s', *x, flush=True)
    log = {'subject': S, 'session': key, 'factors': {}}
    geo, v2r = load_anatomy(S); ses = load_session(S, key, v2r)
    U = unknowns(geo, [(ses['A_anat'], ses['shape'])])
    n, shape, wd, acq = U['n'], ses['shape'], ses['wd'], ses['acq']
    ones = np.ones(n, np.float32)
    say('unknowns', n)

    # ---- per-run data (percent change and raw units)
    bL = np.asarray(nib.load(f'{wd}/runbetaL.nii.gz').dataobj, np.float32)
    bR = np.asarray(nib.load(f'{wd}/runbetaR.nii.gz').dataobj, np.float32)
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    R = bL.shape[-1]
    yrun = (bL + 1j * bR) * mask[..., None]
    raw = yrun * mrun / 100.0
    runs = {'A': list(range(0, R, 2)), 'B': list(range(1, R, 2))}
    subs = {X: (runs[X][0::2], runs[X][1::2]) for X in 'AB'}
    log.update(n_runs=R, runs={k: [r + 1 for r in v] for k, v in runs.items()})

    # ---- native grid: operator, mapping, target maps
    tabs_n, rad_n = SFM.session_tabs(acq)
    K_n = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=shape, affine=ses['A_anat']), tabs_n, rad_n).tocsr()
    W_n, map_n = ribbon_weights(geo, U, ses['A_epi'], ses['A_anat'], f'{wd}/session_mean.nii.gz', od, f'{key}_native')
    dil_n = map_n(K_n @ ones)
    isR = (U['idx'] >= U['nv']['L']).astype(np.float32)

    def xhemi(mp, K):
        """Fraction of each vertex's mapped signal that comes from the other hemisphere's columns (uniform unit field)."""
        fr, fl = mp(K @ isR), mp(K @ (1 - isR))
        return np.where(isR > 0, fl, fr) / np.maximum(fr + fl, 1e-12)
    xh_n = xhemi(map_n, K_n)
    yn = {X: yrun[..., runs[X]].mean(-1).ravel() for X in 'AB'}
    zn = {X: map_n(yn[X]) for X in 'AB'}
    cov_n = {X: (dil_n > 0.5) & (np.abs(zn[X]) > 0) for X in 'AB'}
    sm = smoother_cache(U['Cr'], U['Gr'])
    say('native grid ready; covered', int(cov_n['A'].sum()))
    npz = {'idx': U['idx'].astype(np.int32), 'visual': U['visual'], 'calc': U['calc'], 'xh_n': xh_n.astype(np.float32), 'zn_A': zn['A'].astype(np.complex64),
           'zn_B': zn['B'].astype(np.complex64), 'cov_n_A': cov_n['A'], 'cov_n_B': cov_n['B']}

    for fac in FACTORS:
        t0 = time.time()
        cs = coarse_shape(shape, fac)
        A_epi_c, A_anat_c = coarse_affine(ses['A_epi'], shape, cs), coarse_affine(ses['A_anat'], shape, cs)
        Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]; Ms2 = [M ** 2 for M in Ms]
        # the per-axis operators must reproduce week 2's emulate()
        mimg = np.asarray(ses['img'].dataobj, np.float64)
        e_ref = emulate(mimg, cs); e_new = apply_axes(Ms, mimg)
        emul_err = float(np.abs(e_new - e_ref).max() / np.abs(e_ref).max())
        Em = apply_axes(Ms, mrun)                                        # coarse run means
        Eraw = apply_axes(Ms, raw.real) + 1j * apply_axes(Ms, raw.imag)
        good = Em > 0.05 * np.percentile(Em, 99)
        yc_run = np.where(good, 100.0 * Eraw / np.where(good, Em, 1.0), 0)
        mask_c = (apply_axes(Ms, mask.astype(np.float64)) > 0.9) & good.all(-1)
        tmpl = f'{od}/{key}_x{fac:g}_template.nii.gz'
        nib.save(nib.Nifti1Image(np.where(mask_c, Em.mean(-1), 0).astype(np.float32), A_epi_c), tmpl)
        W_c, map_c = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, od, f'{key}_x{fac:g}')
        csp = voxel_sizes(dict(affine=A_anat_c))
        tabs_c, rad_c = coarse_tabs(acq, [acq['h'], acq['h'], 1.1 * acq['h']], csp)
        K_c = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
        dil_c = map_c(K_c @ ones)
        xh_c = xhemi(map_c, K_c)
        mixed = (xh_c > XH_COARSE) & (xh_n < XH_NATIVE)
        flog = dict(shape=list(cs), voxel_mm=[float(v) for v in csp], emulate_check_rel_err=emul_err, mask_voxels=int(mask_c.sum()),
                    t_setup_s=round(time.time() - t0), directions={})
        say(f'x{fac:g}', 'coarse grid', cs, 'voxel', np.round(csp, 2), 'emulate check', f'{emul_err:.1e}')
        for X, Y in (('A', 'B'), ('B', 'A')):
            t1 = time.time()
            rX, (r1, r2) = runs[X], subs[X]
            ycX = yc_run[..., rX].mean(-1).ravel()
            yc1, yc2 = yc_run[..., r1].mean(-1).ravel(), yc_run[..., r2].mean(-1).ravel()
            # noise of one coarse run: native within-half run-to-run variance propagated through the emulation, calibrated
            vr = (yrun[..., rX].real.var(-1, ddof=1) + yrun[..., rX].imag.var(-1, ddof=1))
            mbar = mrun[..., rX].mean(-1); Ebar = Em[..., rX].mean(-1)
            prop = apply_axes(Ms2, vr * (mbar / 100.0) ** 2) / np.maximum((Ebar / 100.0) ** 2, 1e-12)
            emp = yc_run[..., rX].real.var(-1, ddof=1) + yc_run[..., rX].imag.var(-1, ddof=1)
            cfac = float(emp[mask_c].mean() / max(prop[mask_c].mean(), 1e-30))
            s2run = np.where(mask_c, cfac * prop, 0).ravel()
            s2X = s2run / len(rX)
            zc, z1, z2 = map_c(ycX), map_c(yc1), map_c(yc2)
            cov_c = (dil_c > 0.5) & (np.abs(zc) > 0)
            tset = {'covered': cov_c & cov_n[Y], 'visual': cov_c & cov_n[Y] & U['visual'],
                    'visual_xhemi': cov_c & cov_n[Y] & U['visual'] & mixed}
            # source-side sets for the self-tuning split (the cross-hemisphere set uses the visual choice: too few vertices)
            sset = {'covered': cov_c, 'visual': cov_c & U['visual'], 'visual_xhemi': cov_c & U['visual']}
            target = zn[Y]
            tsc = lambda p: {k: resid_real(target, p, v) for k, v in tset.items()}
            cvs = lambda p1, p2: {k: 0.5 * (resid_real(z2, p1, v) + resid_real(z1, p2, v)) for k, v in sset.items()}
            T, C = {}, {}
            T['standard'] = tsc(zc); C['standard'] = cvs(z1, z2)
            up = lambda y: map_n(np.real(upsample(y.real.reshape(cs), shape)).ravel() + 1j * np.real(upsample(y.imag.reshape(cs), shape)).ravel())
            uX, u1, u2 = up(ycX), up(yc1), up(yc2)
            T['upsample'] = tsc(uX)
            for fw in FW_GRID:
                g = sm(fw)
                T[f'smooth{fw:g}'] = tsc(g(zc)); C[f'smooth{fw:g}'] = cvs(g(z1), g(z2))
                T[f'upsmooth{fw:g}'] = tsc(g(uX))
                C[f'upsmooth{fw:g}'] = {k: 0.5 * (resid_real(u2, g(u1), v) + resid_real(u1, g(u2), v)) for k, v in sset.items()}
            C['upsample'] = {k: 0.5 * (resid_real(u2, u1, v) + resid_real(u1, u2, v)) for k, v in sset.items()}
            # native reference: the source half's own native map
            T['native'] = tsc(zn[X])
            for fw in FW_GRID:
                T[f'native_smooth{fw:g}'] = tsc(sm(fw)(zn[X]))
            # model
            prob = Problem(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi'])
            std_sd = np.sqrt(U['cat']({h: W_c[h].multiply(W_c[h]) @ s2X for h in 'LR'}))
            resp = cov_c & (np.abs(zc) > 3 * np.maximum(std_sd, 1e-12))
            sig0 = float(np.median(np.abs(zc[resp]) / np.maximum(dil_c[resp], 0.3))) if resp.sum() > 50 else float(np.median(np.abs(zc[cov_c])))
            BX, B1, B2 = prob.rhs(ycX), prob.rhs(yc1), prob.rhs(yc2)
            al1, al2 = len(rX) / len(r1), len(rX) / len(r2)                   # sub-half noise = alpha x source-half noise
            X0 = {'h': None, '1': None, '2': None}; its = {}
            for rho, gs in GRID_A:
                tag = f'model_r{rho:g}_s{gs:g}'
                prob.set_prior(rho, gs * sig0); Xh, ith, _ = prob.solve(BX, tol=1e-4, maxit=400, X0=X0['h']); X0['h'] = Xh
                f = Xh[:, 0] + 1j * Xh[:, 1]
                pred = map_n((K_n @ f.real.astype(np.float32)) + 1j * (K_n @ f.imag.astype(np.float32)))
                T[tag] = tsc(pred)
                if (rho, gs) in SAVE_MODEL:
                    npz[f'x{fac:g}_{X}_{tag}'] = pred.astype(np.complex64)
                pr = []
                for sub, Bs, al in (('1', B1, al1), ('2', B2, al2)):
                    prob.set_prior(rho, gs * sig0 / np.sqrt(al)); Xs, _, _ = prob.solve(Bs, tol=1e-4, maxit=400, X0=X0[sub]); X0[sub] = Xs
                    fs = Xs[:, 0] + 1j * Xs[:, 1]
                    pr.append(map_c((K_c @ fs.real.astype(np.float32)) + 1j * (K_c @ fs.imag.astype(np.float32))))
                C[tag] = cvs(pr[0], pr[1])
                its[tag] = int(ith)
            npz.update({f'x{fac:g}_{X}_zc': zc.astype(np.complex64), f'x{fac:g}_{X}_cov_c': cov_c, f'x{fac:g}_mixed': mixed,
                        f'x{fac:g}_xh_c': xh_c.astype(np.float32)})
            flog['directions'][f'{X}->{Y}'] = dict(target=T, cv=C, n={k: int(v.sum()) for k, v in tset.items()},
                                                   n_source={k: int(v.sum()) for k, v in sset.items()}, noise_calibration=cfac,
                                                   sigma0=sig0, n_responsive=int(resp.sum()), iterations=its, t_s=round(time.time() - t1))
            del prob
            say(f'x{fac:g} {X}->{Y}', 'cal', round(cfac, 2), 'visual:', {m: round(T[m]['visual'], 3) for m in
                ('standard', 'smooth4', 'smooth10', 'upsmooth8', 'native', 'model_r12_s0.125', 'model_r24_s0.25', 'model_r6_s1')},
                round(time.time() - t1), 's')
        log['factors'][f'{fac:g}'] = flog
        json.dump(log, open(f'{od}/{S}.{key}.w5_emul_log.json', 'w'), indent=1, default=float)
    np.savez_compressed(f'{od}/{S}.{key}.w5_emul_maps.npz', **npz)
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{od}/{S}.{key}.w5_emul_log.json', 'w'), indent=1, default=float)
    say('done')


if __name__ == '__main__':
    main()
