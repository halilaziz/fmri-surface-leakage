"""Week 6: the week-5 within-session emulation (sf_emul.py) with a correlated-noise likelihood for the model.
Same data, splits, coarse grids, targets and competitors as week 5. The model is fitted twice:
  white  independent voxel noise with the calibrated marginal variance (week 5; small grid around week 5's optimum)
  corr   the same marginal variance times a stationary correlation whose spectrum is estimated from the source half's own run
         residuals on the coarse grid (w6lib.noise_spectrum; no target data)
Extra vertex sets: visual vertices clean at the native grid (< 2% of the map from the other hemisphere), binned by how much of the
coarse map comes from the other hemisphere (0-2, 2-5, 5-10, 10-20, > 20%): the dose-response of unmixing.
Usage (xcp_d container): python3 sf_emul_corr.py SUBJECT SESSION [--factor 1.5|2]
"""
import os, sys, json, time, argparse
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w6lib import *

OUT = f'{ROOT}/results/week6/sf_emul_corr'
GRID_CORR = [(r, g) for r in (3.0, 6.0, 12.0) for g in (0.25, 0.5, 1.0, 2.0, 4.0)]
GRID_WHITE = [(3.0, 4.0), (6.0, 4.0), (6.0, 2.0), (12.0, 2.0)]              # around week 5's self-tuned / LOSO optimum
XH_BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('session'); ap.add_argument('--factor', type=float, default=None)
    a = ap.parse_args(); S, key = a.subject, a.session
    factors = [a.factor] if a.factor else [1.5, 2.0]
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{key}_{"_".join(f"x{f:g}" for f in factors)}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)   # per-job scratch
    T0 = time.time(); say = lambda *x: print(S, key, f'{time.time() - T0:7.0f}s', *x, flush=True)
    tag_out = '_'.join(f'x{f:g}' for f in factors)
    log = {'subject': S, 'session': key, 'factors': {}}
    geo, v2r = load_anatomy(S); ses = load_session(S, key, v2r)
    U = unknowns(geo, [(ses['A_anat'], ses['shape'])])
    n, shape, wd, acq = U['n'], ses['shape'], ses['wd'], ses['acq']
    ones = np.ones(n, np.float32)
    bL = np.asarray(nib.load(f'{wd}/runbetaL.nii.gz').dataobj, np.float32)
    bR = np.asarray(nib.load(f'{wd}/runbetaR.nii.gz').dataobj, np.float32)
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    R = bL.shape[-1]
    yrun = (bL + 1j * bR) * mask[..., None]; raw = yrun * mrun / 100.0
    runs = {'A': list(range(0, R, 2)), 'B': list(range(1, R, 2))}
    subs = {X: (runs[X][0::2], runs[X][1::2]) for X in 'AB'}
    tabs_n, rad_n = SFM.session_tabs(acq)
    K_n = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=shape, affine=ses['A_anat']), tabs_n, rad_n).tocsr()
    W_n, map_n = ribbon_weights(geo, U, ses['A_epi'], ses['A_anat'], f'{wd}/session_mean.nii.gz', tmp, f'{key}_native')
    dil_n = map_n(K_n @ ones)
    isR = (U['idx'] >= U['nv']['L']).astype(np.float32)

    def xhemi(mp, K):
        fr, fl = mp(K @ isR), mp(K @ (1 - isR))
        return np.where(isR > 0, fl, fr) / np.maximum(fr + fl, 1e-12)
    xh_n = xhemi(map_n, K_n)
    yn = {X: yrun[..., runs[X]].mean(-1).ravel() for X in 'AB'}
    zn = {X: map_n(yn[X]) for X in 'AB'}
    cov_n = {X: (dil_n > 0.5) & (np.abs(zn[X]) > 0) for X in 'AB'}
    sm = smoother_cache(U['Cr'], U['Gr'])
    say('native grid ready')

    for fac in factors:
        t0 = time.time()
        cs = coarse_shape(shape, fac)
        A_epi_c, A_anat_c = coarse_affine(ses['A_epi'], shape, cs), coarse_affine(ses['A_anat'], shape, cs)
        Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]; Ms2 = [M ** 2 for M in Ms]
        Em = apply_axes(Ms, mrun)
        Eraw = apply_axes(Ms, raw.real) + 1j * apply_axes(Ms, raw.imag)
        good = Em > 0.05 * np.percentile(Em, 99)
        yc_run = np.where(good, 100.0 * Eraw / np.where(good, Em, 1.0), 0)
        mask_c = (apply_axes(Ms, mask.astype(np.float64)) > 0.9) & good.all(-1)
        tmpl = f'{tmp}/{key}_x{fac:g}_template.nii.gz'
        nib.save(nib.Nifti1Image(np.where(mask_c, Em.mean(-1), 0).astype(np.float32), A_epi_c), tmpl)
        W_c, map_c = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, tmp, f'{key}_x{fac:g}')
        csp = voxel_sizes(dict(affine=A_anat_c))
        tabs_c, rad_c = coarse_tabs(acq, [acq['h'], acq['h'], 1.1 * acq['h']], csp)
        K_c = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
        dil_c = map_c(K_c @ ones)
        xh_c = xhemi(map_c, K_c)
        mixed = (xh_c > 0.10) & (xh_n < 0.02)
        flog = dict(shape=list(cs), voxel_mm=[float(v) for v in csp], directions={})
        for X, Y in (('A', 'B'), ('B', 'A')):
            t1 = time.time()
            rX, (r1, r2) = runs[X], subs[X]
            ycX = yc_run[..., rX].mean(-1).ravel()
            yc1, yc2 = yc_run[..., r1].mean(-1).ravel(), yc_run[..., r2].mean(-1).ravel()
            vr = (yrun[..., rX].real.var(-1, ddof=1) + yrun[..., rX].imag.var(-1, ddof=1))
            mbar = mrun[..., rX].mean(-1); Ebar = Em[..., rX].mean(-1)
            prop = apply_axes(Ms2, vr * (mbar / 100.0) ** 2) / np.maximum((Ebar / 100.0) ** 2, 1e-12)
            emp = yc_run[..., rX].real.var(-1, ddof=1) + yc_run[..., rX].imag.var(-1, ddof=1)
            cfac = float(emp[mask_c].mean() / max(prop[mask_c].mean(), 1e-30))
            s2X = np.where(mask_c, cfac * prop, 0).ravel() / len(rX)
            # noise correlation of this source half (its own runs only)
            Pk, kc, prof = noise_spectrum(unit_residuals(yc_run[..., rX], mask_c), mask_c, csp, cs)
            zc, z1, z2 = map_c(ycX), map_c(yc1), map_c(yc2)
            cov_c = (dil_c > 0.5) & (np.abs(zc) > 0)
            base = cov_c & cov_n[Y]
            tset = {'covered': base, 'visual': base & U['visual'], 'visual_xhemi': base & U['visual'] & mixed}
            for lo, hi in XH_BINS:
                tset[f'xh{int(100 * lo)}'] = base & U['visual'] & (xh_n < 0.02) & (xh_c >= lo) & (xh_c < hi)
            sset = {k: (cov_c if k == 'covered' else cov_c & U['visual']) for k in tset}
            target = zn[Y]
            tsc = lambda p: {k: resid_real(target, p, v) for k, v in tset.items() if v.sum() >= 20}
            cvs = lambda p1, p2: {k: 0.5 * (resid_real(z2, p1, v) + resid_real(z1, p2, v)) for k, v in sset.items()}
            T, C = {}, {}
            T['standard'] = tsc(zc); C['standard'] = cvs(z1, z2)
            up = lambda y: map_n(np.real(upsample(y.real.reshape(cs), shape)).ravel() + 1j * np.real(upsample(y.imag.reshape(cs), shape)).ravel())
            uX, u1, u2 = up(ycX), up(yc1), up(yc2)
            T['upsample'] = tsc(uX); C['upsample'] = cvs(u1, u2)
            for fw in FW_GRID:
                g = sm(fw)
                T[f'smooth{fw:g}'] = tsc(g(zc)); C[f'smooth{fw:g}'] = cvs(g(z1), g(z2))
                T[f'upsmooth{fw:g}'] = tsc(g(uX)); C[f'upsmooth{fw:g}'] = cvs(g(u1), g(u2))
            T['native'] = tsc(zn[X])
            std_sd = np.sqrt(U['cat']({h: W_c[h].multiply(W_c[h]) @ s2X for h in 'LR'}))
            resp = cov_c & (np.abs(zc) > 3 * np.maximum(std_sd, 1e-12))
            sig0 = float(np.median(np.abs(zc[resp]) / np.maximum(dil_c[resp], 0.3))) if resp.sum() > 50 else float(np.median(np.abs(zc[cov_c])))
            al1, al2 = len(rX) / len(r1), len(rX) / len(r2)
            its = {}
            for lab, grid, mk in (('white', GRID_WHITE, lambda: Problem(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi'])),
                                  ('corr', GRID_CORR, lambda: ProblemCorr(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi'], cs, Pk))):
                prob = mk(); BX, B1, B2 = prob.rhs(ycX), prob.rhs(yc1), prob.rhs(yc2)
                X0 = {'h': None, '1': None, '2': None}
                for rho, gs in grid:
                    tag = f'{lab}_r{rho:g}_s{gs:g}'
                    prob.set_prior(rho, gs * sig0); Xh, ith, _ = prob.solve(BX, tol=1e-4, maxit=500, X0=X0['h']); X0['h'] = Xh
                    f = Xh[:, 0] + 1j * Xh[:, 1]
                    T[tag] = tsc(map_n((K_n @ f.real.astype(np.float32)) + 1j * (K_n @ f.imag.astype(np.float32))))
                    pr = []
                    for sub, Bs, al in (('1', B1, al1), ('2', B2, al2)):
                        prob.set_prior(rho, gs * sig0 / np.sqrt(al)); Xs, _, _ = prob.solve(Bs, tol=1e-4, maxit=500, X0=X0[sub]); X0[sub] = Xs
                        fs = Xs[:, 0] + 1j * Xs[:, 1]
                        pr.append(map_c((K_c @ fs.real.astype(np.float32)) + 1j * (K_c @ fs.imag.astype(np.float32))))
                    C[tag] = cvs(pr[0], pr[1]); its[tag] = int(ith)
                    if (rho, gs) == grid[0]:
                        say(f'x{fac:g} {X}->{Y}', lab, tag, 'iterations', ith, 'visual', round(T[tag]['visual'], 3), round(time.time() - t1), 's')
                del prob
            flog['directions'][f'{X}->{Y}'] = dict(target=T, cv=C, n={k: int(v.sum()) for k, v in tset.items()}, noise_calibration=cfac,
                                                   sigma0=sig0, iterations=its, spectrum=dict(k=kc.tolist(), power=prof.tolist()),
                                                   t_s=round(time.time() - t1))
            bw = min((k for k in T if k.startswith('white_')), key=lambda k: T[k]['visual'])
            bc = min((k for k in T if k.startswith('corr_')), key=lambda k: T[k]['visual'])
            bu = min((k for k in T if k.startswith('upsmooth') or k == 'upsample'), key=lambda k: T[k]['visual'])
            say(f'x{fac:g} {X}->{Y}', 'visual: best upsampling', bu, round(T[bu]['visual'], 3), '| white', bw, round(T[bw]['visual'], 3),
                '| corr', bc, round(T[bc]['visual'], 3), '| native', round(T['native']['visual'], 3), '| iters', its.get(bc), round(time.time() - t1), 's')
        log['factors'][f'{fac:g}'] = flog
        json.dump(log, open(f'{od}/{S}.{key}.{tag_out}.w6_log.json', 'w'), indent=1, default=float)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    say('done')


if __name__ == '__main__':
    main()
