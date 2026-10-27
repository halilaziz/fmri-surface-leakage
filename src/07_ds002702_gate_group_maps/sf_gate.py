"""Week 7, StudyForrest within-session emulation (weeks 5-6 design): per-vertex predictions for the leak-gated estimator.
For one subject / native 7T session / coarsening factor and both directions (odd -> even runs, even -> odd), saves:
  target (other half's native standard map), competitor predictions (upsampling, upsampling + 2 mm), model predictions for the
  settings week 6's leave-one-subject-out selection chose (correlated noise; white noise for reference), and gate variables:
  far leak of standard mapping (week-2 definition: other hemisphere or > 15 mm geodesic on the sphere) and the other-hemisphere
  share, at the native and the coarse grid. summarize_w7.py builds the gated estimator and chooses its threshold.
Usage (xcp_d container): python3 sf_gate.py SUBJECT SESSION --factor 1.5|2
"""
import os, sys, json, time, argparse, shutil
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '06_noise_dose_response'))
from w6lib import *

OUT = f'{ROOT}/results/week7/sf_gate'
MODELS = [('corr', 6.0, 2.0), ('corr', 12.0, 2.0), ('corr', 12.0, 4.0), ('white', 6.0, 2.0), ('white', 6.0, 4.0)]
GEO_THR = 15.0


def far_leak(W, K, U, geo, sph):
    """Per unknown vertex: share of the standard-mapping readout of a unit field that comes from far cortex (other hemisphere, or
    > 15 mm geodesic, measured on the registration sphere scaled to the cortical area; week 2) and from the other hemisphere."""
    idx, nL = U['idx'], U['nv']['L']
    colh = (idx >= nL); colloc = np.where(colh, idx - nL, idx)
    su, fs = {}, {}
    for h in 'LR':
        s = sph[h]; Rs = np.linalg.norm(s, axis=1).mean(); su[h] = s / np.linalg.norm(s, axis=1, keepdims=True)
        a, _ = vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])
        fs[h] = Rs * np.sqrt(a[geo[h]['roi']].sum() / (4 * np.pi * Rs ** 2))
    far_all, oh_all = np.zeros(len(idx)), np.zeros(len(idx))
    for h in 'LR':
        sel = np.flatnonzero((idx >= nL) == (h == 'R')); rows = colloc[sel]
        for c0 in range(0, len(rows), 3000):
            r = rows[c0:c0 + 3000]
            A = (W[h][r] @ K).tocoo()
            val = A.data.astype(np.float64); j = A.col
            other = colh[j] != (h == 'R')
            cosang = np.einsum('nd,nd->n', su[h][r[A.row]], su[h][np.where(other, 0, colloc[j])])
            far = other | (fs[h] * np.arccos(np.clip(cosang, -1, 1)) > GEO_THR)
            tot = np.bincount(A.row, val, minlength=len(r))
            far_all[sel[c0:c0 + 3000]] = np.bincount(A.row, val * far, minlength=len(r)) / np.maximum(np.abs(tot), 1e-12)
            oh_all[sel[c0:c0 + 3000]] = np.bincount(A.row, val * other, minlength=len(r)) / np.maximum(np.abs(tot), 1e-12)
    return far_all, oh_all


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('session'); ap.add_argument('--factor', type=float, required=True)
    a = ap.parse_args(); S, key, fac = a.subject, a.session, a.factor
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{key}_x{fac:g}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, key, f'x{fac:g}', f'{time.time() - T0:7.0f}s', *x, flush=True)
    geo, v2r = load_anatomy(S); ses = load_session(S, key, v2r)
    sph = {h: nib.freesurfer.read_geometry(f'{SF}/freesurfer/{S}/surf/{fh}.sphere')[0] for h, fh in (('L', 'lh'), ('R', 'rh'))}
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
    tabs_n, rad_n = SFM.session_tabs(acq)
    K_n = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=shape, affine=ses['A_anat']), tabs_n, rad_n).tocsr()
    W_n, map_n = ribbon_weights(geo, U, ses['A_epi'], ses['A_anat'], f'{wd}/session_mean.nii.gz', tmp, f'{key}_native')
    dil_n = map_n(K_n @ ones)
    far_n, xh_n = far_leak(W_n, K_n, U, geo, sph)
    zn = {X: map_n(yrun[..., runs[X]].mean(-1).ravel()) for X in 'AB'}
    cov_n = {X: (dil_n > 0.5) & (np.abs(zn[X]) > 0) for X in 'AB'}
    say('native grid and far leak ready')
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
    far_c, xh_c = far_leak(W_c, K_c, U, geo, sph)
    say('coarse grid and far leak ready; median far leak native / coarse (visual):',
        round(float(np.median(far_n[U['visual']])), 3), round(float(np.median(far_c[U['visual']])), 3))
    npz = dict(idx=U['idx'].astype(np.int32), visual=U['visual'], far_n=far_n.astype(np.float32), far_c=far_c.astype(np.float32),
               xh_n=xh_n.astype(np.float32), xh_c=xh_c.astype(np.float32), dil_n=dil_n.astype(np.float32), dil_c=dil_c.astype(np.float32))
    log = dict(subject=S, session=key, factor=fac, voxel_mm=[float(v) for v in csp], models=[list(m) for m in MODELS], directions={})
    for X, Y in (('A', 'B'), ('B', 'A')):
        t1 = time.time(); rX = runs[X]
        ycX = yc_run[..., rX].mean(-1).ravel()
        vr = (yrun[..., rX].real.var(-1, ddof=1) + yrun[..., rX].imag.var(-1, ddof=1))
        mbar = mrun[..., rX].mean(-1); Ebar = Em[..., rX].mean(-1)
        prop = apply_axes(Ms2, vr * (mbar / 100.0) ** 2) / np.maximum((Ebar / 100.0) ** 2, 1e-12)
        emp = yc_run[..., rX].real.var(-1, ddof=1) + yc_run[..., rX].imag.var(-1, ddof=1)
        cfac = float(emp[mask_c].mean() / max(prop[mask_c].mean(), 1e-30))
        s2X = np.where(mask_c, cfac * prop, 0).ravel() / len(rX)
        Pk, _, _ = noise_spectrum(unit_residuals(yc_run[..., rX], mask_c), mask_c, csp, cs)
        zc = map_c(ycX); cov_c = (dil_c > 0.5) & (np.abs(zc) > 0)
        up = map_n(np.real(upsample(ycX.real.reshape(cs), shape)).ravel() + 1j * np.real(upsample(ycX.imag.reshape(cs), shape)).ravel())
        std_sd = np.sqrt(U['cat']({h: W_c[h].multiply(W_c[h]) @ s2X for h in 'LR'}))
        resp = cov_c & (np.abs(zc) > 3 * np.maximum(std_sd, 1e-12))
        sig0 = float(np.median(np.abs(zc[resp]) / np.maximum(dil_c[resp], 0.3))) if resp.sum() > 50 else float(np.median(np.abs(zc[cov_c])))
        d = f'{X}{Y}'
        npz[f'{d}_target'] = zn[Y].astype(np.complex64); npz[f'{d}_base'] = cov_c & cov_n[Y]
        npz[f'{d}_upsample'] = up.astype(np.complex64)
        npz[f'{d}_upsmooth2'] = smoother_cache(U['Cr'], U['Gr'])(2.0)(up).astype(np.complex64)
        its = {}
        for lab in ('white', 'corr'):
            ms = [(r_, s_) for l_, r_, s_ in MODELS if l_ == lab]
            prob = (Problem(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi']) if lab == 'white'
                    else ProblemCorr(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi'], cs, Pk))
            B = prob.rhs(ycX); X0 = None
            for rho, gs in ms:
                prob.set_prior(rho, gs * sig0); Xh, it, _ = prob.solve(B, tol=1e-4, maxit=500, X0=X0); X0 = Xh
                f = Xh[:, 0] + 1j * Xh[:, 1]
                npz[f'{d}_{lab}_r{rho:g}_s{gs:g}'] = map_n((K_n @ f.real.astype(np.float32)) + 1j * (K_n @ f.imag.astype(np.float32))).astype(np.complex64)
                its[f'{lab}_r{rho:g}_s{gs:g}'] = int(it)
            del prob
        log['directions'][d] = dict(noise_calibration=cfac, sigma0=sig0, iterations=its, t_s=round(time.time() - t1))
        say(X, '->', Y, its, round(time.time() - t1), 's')
    np.savez_compressed(f'{od}/{S}.{key}.x{fac:g}.w7_gate.npz', **npz)
    json.dump(log, open(f'{od}/{S}.{key}.x{fac:g}.w7_gate_log.json', 'w'), indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    say('done')


if __name__ == '__main__':
    main()
