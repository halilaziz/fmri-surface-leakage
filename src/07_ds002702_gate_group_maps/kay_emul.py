"""Week 7, ds002702 (Kay et al. 7T 0.8 mm eccentricity rings): within-session emulation, weeks 5-7 design, on a third dataset.
One subject, the 7T 0.8 mm session, one coarsening factor (x2 -> 1.6 mm, x3 -> 2.4 mm = the 3T session's voxel), both directions
(odd -> even runs, even -> odd). Data are the six ring-condition betas (real maps; each is scored with its own best scaling).
Saves per-vertex predictions for the gated estimator: competitors (upsampling, + 2 / 4 mm smoothing; coarse standard map + smoothing),
the model (correlated and white noise, a small prior grid), targets, and gate variables (far leak and other-hemisphere share of
standard mapping at the native and coarse grid, week-2 definition).
Usage (xcp_d container): python3 kay_emul.py SUBJECT --factor 2|3 [--session ring7T]
"""
import os, sys, json, time, argparse, shutil, glob
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '06_noise_dose_response'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w6lib import *
from sf_gate import far_leak

G_ = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
DATA, FSD, PREPD, OUT = f'{G_}/data/ds002702', f'{G_}/data/ds002702_freesurfer', f'{G_}/results/kay/prep', f'{G_}/results/kay/emul'
CONDS = [f'ecc{k}' for k in range(1, 7)]
# prior grids (range mm, s.d. scale). The first fits with the week-7 StudyForrest grid (corr 3-12 mm x 2-4, white 6 mm x 2-4) had
# their optimum at its edge (sub-01: x3 at corr 12 mm / 2, x1.3125 at white 6 mm / 2), so both grids extend to stronger priors;
# at x1.3125 the white-noise model was clearly better than the correlated one and gets the denser grid
MODELS_FINE = [('white', r, s) for r in (3.0, 6.0, 12.0) for s in (2.0, 1.0, 0.5)] + [('corr', r, s) for r in (6.0, 12.0) for s in (4.0, 2.0, 1.0)]
MODELS_COARSE = [('corr', r, s) for r in (6.0, 12.0, 24.0) for s in (4.0, 2.0, 1.0, 0.5)] + [('white', 12.0, 2.0), ('white', 12.0, 1.0)]


def models_for(fac):
    return MODELS_FINE if fac < 1.75 else MODELS_COARSE


T2S = {'7T': 28.0, '3T': 45.0}


def load_anatomy_fs(fs):
    orig = nib.load(f'{fs}/mri/orig.mgz'); v2r = orig.affine; tkr2scan = v2r @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    geo, sph = {}, {}
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        W, F = nib.freesurfer.read_geometry(f'{fs}/surf/{fh}.white'); Pv, _ = nib.freesurfer.read_geometry(f'{fs}/surf/{fh}.pial')
        lab, _, names = nib.freesurfer.read_annot(f'{fs}/label/{fh}.aparc.annot')
        nm = np.array([names[l] if l >= 0 else b'unknown' for l in lab])
        W, Pv = SFM.hom(tkr2scan, W), SFM.hom(tkr2scan, Pv)
        geo[h] = dict(white=W, pial=Pv, mid=0.5 * (W + Pv), faces=F.astype(np.int64),
                      roi=~np.isin(nm, [b'unknown', b'corpuscallosum']), visual=np.isin(nm, SFM.VISUAL), calc=nm == b'pericalcarine')
        sph[h] = nib.freesurfer.read_geometry(f'{fs}/surf/{fh}.sphere')[0]
    return geo, sph


def acq_from_json(js, field):
    h = float(js['SliceThickness']); pe = {'i': 0, 'j': 1}[js['PhaseEncodingDirection'][0]]
    return dict(h=h, pe_axis=pe, esp_eff_ms=1000 * float(js['EffectiveEchoSpacing']), fov_pe_mm=int(js['AcquisitionMatrixPE']) * h,
                t2s_ms=T2S[field], slice_edge=0.15, taper=0.3, sigma_interp=0.3 * h)


def native_tabs(acq):
    """Week-4 session_tabs without the slice gap (these sessions have contiguous slices)."""
    rad = SUPPORT_VOX['1.6']; sp_ = [acq['h']] * 3
    return [kernel_table(d, 1.0, acq=acq, trunc_mm=(rad[d] + 0.5) * sp_[d], R_mm=(rad[d] + 1) * sp_[d]) for d in range(3)], rad


def resid_multi(T, P, sel):
    """Residual energy fraction pooled over maps (columns), each map with its own best scaling."""
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def rhs_multi(prob, Y):
    Yr = Y[prob.rows].astype(np.float64)
    Z = prob.sqD[:, None] * prob.Cinv(prob.sqD[:, None] * Yr) if hasattr(prob, 'Cinv') else prob.D[:, None].astype(np.float64) * Yr
    return np.asarray(prob.Kt @ Z.astype(np.float32), np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--factor', type=float, required=True)
    ap.add_argument('--session', default='ring7T')
    a = ap.parse_args(); S, fac, ses = a.subject, a.factor, a.session
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{ses}_x{fac:g}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, ses, f'x{fac:g}', f'{time.time() - T0:7.0f}s', *x, flush=True)
    wd = f'{PREPD}/{S}/{ses}'
    geo, sph = load_anatomy_fs(f"{FSD}/subj{S.split('-')[1]}")
    mimg = nib.load(f'{wd}/session_mean.nii.gz'); A_epi = mimg.affine; shape = mimg.shape[:3]
    A_anat = np.loadtxt(f'{wd}/epi2anat_vox2ras.txt')
    js = json.load(open(sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/func/*task-ring_run-01_bold.json'))[0]))
    acq = acq_from_json(js, '3T' if js['MagneticFieldStrength'] < 5 else '7T')
    U = unknowns(geo, [(A_anat, shape)])
    n = U['n']; ones = np.ones(n, np.float32)
    say('unknowns', n, 'acq', {k: round(v, 3) if isinstance(v, float) else v for k, v in acq.items()})
    Y = np.stack([np.asarray(nib.load(f'{wd}/runbeta_{c}.nii.gz').dataobj, np.float32) for c in CONDS], 3)   # x y z cond run
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    Y = Y * mask[..., None, None]; R = Y.shape[-1]
    runs = {'A': list(range(0, R, 2)), 'B': list(range(1, R, 2))}
    tabs_n, rad_n = native_tabs(acq)
    K_n = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=shape, affine=A_anat), tabs_n, rad_n).tocsr()
    W_n, map_n = ribbon_weights(geo, U, A_epi, A_anat, f'{wd}/session_mean.nii.gz', tmp, f'{ses}_native')
    dil_n = map_n(K_n @ ones)
    far_n, xh_n = far_leak(W_n, K_n, U, geo, sph)
    zn = {X: map_n(Y[..., runs[X]].mean(-1).reshape(-1, len(CONDS))) for X in 'AB'}
    cov_n = {X: (dil_n > 0.5) & (np.abs(zn[X]).sum(1) > 0) for X in 'AB'}
    say('native grid ready; median far leak (visual)', round(float(np.median(far_n[U['visual']])), 3))
    cs = coarse_shape(shape, fac)
    A_epi_c, A_anat_c = coarse_affine(A_epi, shape, cs), coarse_affine(A_anat, shape, cs)
    Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]; Ms2 = [M ** 2 for M in Ms]
    Em = apply_axes(Ms, mrun); good = Em > 0.05 * np.percentile(Em, 99)
    raw = Y * mrun[..., None, :] / 100.0
    Yc = np.where(good[..., None, :], 100.0 * apply_axes(Ms, raw) / np.where(good, Em, 1.0)[..., None, :], 0)   # coarse x y z cond run
    mask_c = (apply_axes(Ms, mask.astype(np.float64)) > 0.9) & good.all(-1)
    tmpl = f'{tmp}/{ses}_x{fac:g}_template.nii.gz'
    nib.save(nib.Nifti1Image(np.where(mask_c, Em.mean(-1), 0).astype(np.float32), A_epi_c), tmpl)
    W_c, map_c = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, tmp, f'{ses}_x{fac:g}')
    csp = voxel_sizes(dict(affine=A_anat_c))
    tabs_c, rad_c = coarse_tabs(acq, [acq['h']] * 3, csp)
    K_c = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
    dil_c = map_c(K_c @ ones)
    far_c, xh_c = far_leak(W_c, K_c, U, geo, sph)
    say('coarse grid', cs, np.round(csp, 2), 'ready; median far leak (visual)', round(float(np.median(far_c[U['visual']])), 3))
    npz = dict(idx=U['idx'].astype(np.int32), visual=U['visual'], calc=U['calc'], far_n=far_n.astype(np.float32), far_c=far_c.astype(np.float32),
               xh_n=xh_n.astype(np.float32), xh_c=xh_c.astype(np.float32))
    sm = smoother_cache(U['Cr'], U['Gr'])
    smm = lambda fw, X: np.stack([sm(fw)(X[:, j]) for j in range(X.shape[1])], 1)
    MODELS = models_for(fac)
    log = dict(subject=S, session=ses, factor=fac, voxel_mm=[float(v) for v in csp], acq=acq, models=[list(m) for m in MODELS], directions={})
    for X, Yh in (('A', 'B'), ('B', 'A')):
        t1 = time.time(); rX = runs[X]
        ycX = Yc[..., rX].mean(-1).reshape(-1, len(CONDS))
        # noise: native within-half run-to-run variance (mean over conditions) propagated, calibrated on the coarse runs
        vr = Y[..., rX].var(-1, ddof=1).mean(3)
        mbar = mrun[..., rX].mean(-1); Ebar = Em[..., rX].mean(-1)
        prop = apply_axes(Ms2, vr * (mbar / 100.0) ** 2) / np.maximum((Ebar / 100.0) ** 2, 1e-12)
        emp = Yc[..., rX].var(-1, ddof=1).mean(3)
        cfac = float(emp[mask_c].mean() / max(prop[mask_c].mean(), 1e-30))
        s2X = np.where(mask_c, cfac * prop, 0).ravel() / len(rX)
        res_maps = []
        for j in range(len(CONDS)):
            Yj = Yc[..., j, rX]; r_ = Yj - Yj.mean(-1, keepdims=True); sd = np.sqrt((r_ ** 2).sum(-1) / (len(rX) - 1))
            ok = mask_c & (sd > 0)
            res_maps += [np.where(ok, r_[..., i] / np.where(ok, sd, 1.0), 0.0) for i in range(len(rX))]
        Pk, _, _ = noise_spectrum(res_maps, mask_c, csp, cs)
        zc = map_c(ycX); cov_c = (dil_c > 0.5) & (np.abs(zc).sum(1) > 0)
        up = map_n(np.stack([np.real(upsample(ycX[:, j].reshape(cs), shape)).ravel() for j in range(len(CONDS))], 1))
        std_sd = np.sqrt(U['cat']({h: W_c[h].multiply(W_c[h]) @ s2X for h in 'LR'}))
        amp = np.sqrt((zc ** 2).mean(1)); resp = cov_c & (amp > 3 * np.maximum(std_sd, 1e-12))
        sig0 = float(np.median(amp[resp] / np.maximum(dil_c[resp], 0.3))) if resp.sum() > 50 else float(np.median(amp[cov_c]))
        d = f'{X}{Yh}'
        npz[f'{d}_target'] = zn[Yh].astype(np.float32); npz[f'{d}_base'] = cov_c & cov_n[Yh]
        npz[f'{d}_native'] = zn[X].astype(np.float32)
        npz[f'{d}_upsample'] = up.astype(np.float32)
        for fw in (2.0, 4.0):
            npz[f'{d}_upsmooth{fw:g}'] = smm(fw, up).astype(np.float32)
        for fw in (4.0, 8.0):
            npz[f'{d}_smooth{fw:g}'] = smm(fw, zc).astype(np.float32)
        its = {}
        for lab in ('white', 'corr'):
            ms = [(r_, s_) for l_, r_, s_ in MODELS if l_ == lab]
            prob = (Problem(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi']) if lab == 'white'
                    else ProblemCorr(K_c, mask_c.ravel(), s2X, U['Cr'], U['Gr'], U['nroi'], cs, Pk))
            if not hasattr(prob, 'sqD'):
                prob.sqD = np.sqrt(prob.D).astype(np.float64)
            B = rhs_multi(prob, ycX); X0 = None
            for rho, gs in ms:
                prob.set_prior(rho, gs * sig0); Xh, it, _ = prob.solve(B, tol=1e-4, maxit=500, X0=X0); X0 = Xh
                npz[f'{d}_{lab}_r{rho:g}_s{gs:g}'] = map_n(np.asarray(K_n @ Xh.astype(np.float32))).astype(np.float32)
                its[f'{lab}_r{rho:g}_s{gs:g}'] = int(it)
            del prob
        sel = npz[f'{d}_base'] & U['visual']
        quick = {k: round(resid_multi(npz[f'{d}_target'], npz[f'{d}_{k}'], sel), 3) for k in
                 ['native', 'upsample', 'upsmooth2', 'smooth4'] + [f'{l}_r{r:g}_s{s:g}' for l, r, s in MODELS]}
        log['directions'][d] = dict(noise_calibration=cfac, sigma0=sig0, iterations=its, visual_quick=quick, n_visual=int(sel.sum()),
                                    t_s=round(time.time() - t1))
        say(X, '->', Yh, 'cal', round(cfac, 2), 'visual', quick, round(time.time() - t1), 's')
    np.savez_compressed(f'{od}/{S}.{ses}.x{fac:g}.kay_emul.npz', **npz)
    json.dump(log, open(f'{od}/{S}.{ses}.x{fac:g}.kay_emul_log.json', 'w'), indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    say('done')


if __name__ == '__main__':
    main()
