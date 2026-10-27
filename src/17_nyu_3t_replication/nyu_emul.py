"""Week 17, NYU Retinotopy (OpenNeuro ds003787): within-session emulation of one subject, as PLAN_NYU.md.
Polar-angle maps: per run and grid, a GLM with the aperture's coverage of the stimulated field weighted by 1, cos(theta) and
sin(theta) (HRF-convolved), the derivative of the first, Legendre drifts 0-3 and six motion parameters; z = b_cos + i b_sin in
percent signal change, noise variance from the residuals. Halves = odd / even kept runs. Each corrected series is coarsened
frame by frame to 3.0 mm (x1.5) and 4.0 mm (x2.0) with the HCP k-space crop + Tukey window and the GLM is rerun.
Prediction of the other half's native 2.0 mm standard map from the coarsened half, both directions: standard mapping, + surface
smoothing, k-space upsampling, + smoothing, the model (white-noise likelihood, Matern prior) on a grid of settings.
Scores: the HCP residual (best complex scaling) per vertex set and per dose bin (far leak, CWL-M), averaged over directions.
Writes GEO_CONST/results/nyu/emul/SUBJECT.nyu_scores.json (and with --save-maps an npz of selected maps).
Usage (xcp_d container): python3 nyu_emul.py SUBJECT [--settings '3.0=12:0.125;4.0=24:0.25'] [--save-maps]
"""
import os, sys, json, time, argparse, shutil, glob
import numpy as np, nibabel as nib
from scipy.stats import gamma
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '07_ds002702_gate_group_maps')); sys.path.insert(0, os.path.join(HERE, '..', '08_contamination'))
from kay_emul import *                      # w6lib -> w5lib -> w3lib / w2lib, load_anatomy_fs, acq_from_json, native_tabs
from sf_gate import far_leak
from cwl_lib import sphere_frames, far_contamination, cwl, split_complex

G_ = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
DATA = f'{G_}/data/ds003787'; FSD = f'{DATA}/derivatives/freesurfer'; PREPD = f'{G_}/results/nyu/prep'; OUT = f'{G_}/results/nyu/emul'
SES = 'ses-nyu3t01'
FACTORS = {'3.0': 1.5, '4.0': 2.0}
FWHMS = (2, 4, 6, 8, 10, 12, 14, 16)
# DEVIATIONS_NYU.md 1: extended from r {6,12,24,48} x s {0.0625..0.5} because the first tuning subject's optimum lay at the planned
# grid's weakest prior and shortest range (as for ds002702, week 7)
GRID = [(r, s) for r in (3.0, 6.0, 12.0, 24.0, 48.0) for s in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)]
R2_THR = 0.2
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, np.inf)]
BIN_NAMES = ['0-0.02', '0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2-inf']
MIN_N = 30


def hrf(tr, n=32):
    dt = tr / 16; t = np.arange(0, n, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return h / h.sum(), dt


def stim_regressors(ap, tr):
    """ap: (nx, ny, T) binary aperture, one frame per volume. Returns (T, 4): coverage weighted by 1, cos, sin of the polar
    angle about the centre (HRF-convolved, sampled at mid-volume) and the temporal derivative of the first."""
    nx, ny, T = ap.shape
    field = ap.any(2); N = float(field.sum()); c = (nx - 1) / 2.0
    ii, jj = np.mgrid[0:nx, 0:ny].astype(float)
    th = np.arctan2(-(ii - c), jj - c)
    a = ap.reshape(-1, T).astype(np.float64)
    raw = [a.sum(0) / N, (a * np.cos(th).ravel()[:, None]).sum(0) / N, (a * np.sin(th).ravel()[:, None]).sum(0) / N]
    h, dt = hrf(tr); up = int(round(tr / dt))

    def conv(x):
        fine = np.repeat(x, up); y = np.convolve(fine, h)[:len(fine)]
        return y[np.arange(T) * up + up // 2]
    R = np.stack([conv(x) for x in raw], 1)
    return np.column_stack([R, np.gradient(R[:, 0])])


def design(stim, par):
    T = stim.shape[0]; x = np.linspace(-1, 1, T)
    leg = np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(4)], 1)
    return np.concatenate([stim, leg] + ([par[:T]] if par is not None else []), 1)


def glm_z(Y, X):
    """Y: (V, T); X: (T, P), columns 1, 2 = cos, sin regressors. Returns complex z (percent signal change), its noise variance
    (complex power), and the voxel mean."""
    XtXi = np.linalg.inv(X.T @ X); Y = Y.astype(np.float64)
    B = Y @ (X @ XtXi)                                       # (V, P)
    res = Y - B @ X.T
    s2 = (res ** 2).sum(1) / (X.shape[0] - X.shape[1])
    mean = np.maximum(Y.mean(1), 1e-6)
    z = 100.0 * (B[:, 1] + 1j * B[:, 2]) / mean
    vz = 1e4 * s2 * (XtXi[1, 1] + XtXi[2, 2]) / mean ** 2
    return z, vz, mean


def r2_stim(Ym, X_full, X_red):
    """R^2 of the stimulus regressors over the drift-only model, per row of Ym (V, T)."""
    def ssr(X):
        B = np.linalg.lstsq(X, Ym.T, rcond=None)[0]
        return ((Ym.T - X @ B) ** 2).sum(0)
    return 1.0 - ssr(X_full) / np.maximum(ssr(X_red), 1e-30)


def far_contamination_oh(W, K, idx, nL, su, fs, Xs, chunk=3000):
    """cwl_lib.far_contamination plus the other-hemisphere share of the readout (sf_gate.far_leak's second output), in one pass
    over the rows of W K (the coarse grid's far leak, contamination and other-hemisphere share; identical definitions)."""
    n = len(idx); colh = idx >= nL; colloc = np.where(colh, idx - nL, idx)
    out = {k: np.zeros((n, X.shape[1])) for k, X in Xs.items()}; far_frac = np.zeros(n); oh_frac = np.zeros(n)
    for h in 'LR':
        sel = np.flatnonzero(colh == (h == 'R')); rows = colloc[sel]
        for c0 in range(0, len(rows), chunk):
            r = rows[c0:c0 + chunk]; sv = sel[c0:c0 + chunk]; nr = len(r)
            A = (W[h][r] @ K).tocoo()
            val = A.data.astype(np.float64); j = A.col
            other = colh[j] != (h == 'R')
            cosang = np.einsum('nd,nd->n', su[h][r[A.row]], su[h][np.where(other, 0, colloc[j])])
            far = other | (fs[h] * np.arccos(np.clip(cosang, -1, 1)) > 15.0)
            tot = np.bincount(A.row, val, minlength=nr); den = np.maximum(np.abs(tot), 1e-12)
            vf = val * far; wf = np.bincount(A.row, vf, minlength=nr)
            far_frac[sv] = wf / den; oh_frac[sv] = np.bincount(A.row, val * other, minlength=nr) / den
            for k, X in Xs.items():
                for q in range(X.shape[1]):
                    s_ = np.bincount(A.row, vf * X[j, q], minlength=nr)
                    out[k][sv, q] = (s_ - wf * X[sv, q]) / den
    return out, far_frac, oh_frac


def resid_c(z, p):
    c = np.vdot(p, z) / max(np.vdot(p, p).real, 1e-30)
    return float(np.sum(np.abs(z - c * p) ** 2) / max(np.sum(np.abs(z) ** 2), 1e-30))


def angle_err(z, p):
    """Mean absolute angular difference (degrees) between the phases of the best-scaled prediction and the target."""
    c = np.vdot(p, z) / max(np.vdot(p, p).real, 1e-30)
    return float(np.degrees(np.mean(np.abs(np.angle(z * np.conj(c * p))))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--settings', default='')
    ap.add_argument('--save-maps', action='store_true')
    ap.add_argument('--gate', action='store_true', help='PLAN_NYU_GATE.md: CWL-P, the CWL-P-gated estimator, field statistics')
    ap.add_argument('--comp', default='upsmooth2', help='competitor of the gated estimator')
    ap.add_argument('--out', default='')
    ap.add_argument('--atlas', action='store_true', help='PLAN_ATLAS.md: contamination with the Benson-template response (results/nyu/atlas/S.benson14.npz)')
    a = ap.parse_args(); S = a.subject
    global OUT
    if a.out:
        OUT = a.out
    os.makedirs(OUT, exist_ok=True); tmp = f'{OUT}/tmp_{S}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)
    grid_settings = {g: GRID for g in FACTORS}
    if a.settings:
        grid_settings = {kv.split('=')[0]: [tuple(float(x) for x in s.split(':')) for s in kv.split('=')[1].split(',')] for kv in a.settings.split(';')}
    wd = f'{PREPD}/{S}'; qc = json.load(open(f'{wd}/{S}_prep_qc.json'))
    if not qc.get('included', False):
        say('excluded at preprocessing', qc.get('runs_kept'), qc.get('bbr_costs')); shutil.rmtree(tmp, ignore_errors=True)
        json.dump(dict(subject=S, excluded=True, runs_kept=qc.get('runs_kept'), bbr_costs=qc.get('bbr_costs'), t_total_s=0),
                  open(f'{OUT}/{S}.nyu_scores.json', 'w'), indent=1, default=float)
        return
    keep = qc['runs_kept']
    geo, sph = load_anatomy_fs(f'{FSD}/{S}')
    mimg = nib.load(f'{wd}/session_mean.nii.gz'); A_epi = mimg.affine; shape = mimg.shape[:3]
    A_anat = np.loadtxt(f'{wd}/epi2anat_vox2ras.txt')
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    js = json.load(open(glob.glob(f'{DATA}/{S}/{SES}/func/{S}_{SES}_task-prf*_run-{keep[0]}_bold.json')[0]))
    acq = acq_from_json(js, '3T'); tr = float(js['RepetitionTime'])
    apert = np.asarray(nib.load(f'{DATA}/derivatives/stimulus_apertures/{S}/{SES}/{S}_{SES}_task-prf_apertures.nii.gz').dataobj) > 0
    stim = stim_regressors(apert, tr)
    U = unknowns(geo, [(A_anat, shape)]); n = U['n']; ones = np.ones(n, np.float32)
    say('unknowns', n, 'runs', keep, 'acq', {k: round(v, 3) if isinstance(v, float) else v for k, v in acq.items()})
    # ---- native grid: operator, ribbon weights, far leak
    tabs_n, rad_n = native_tabs(acq)
    K_n = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=shape, affine=A_anat), tabs_n, rad_n).tocsr()
    W_n, map_n = ribbon_weights(geo, U, A_epi, A_anat, f'{wd}/session_mean.nii.gz', tmp, 'native')
    dil_n = map_n(K_n @ ones); far_n, xh_n = far_leak(W_n, K_n, U, geo, sph)
    area = U['cat']({h: vertex_areas_normals(geo[h]['mid'], geo[h]['faces'])[0] for h in 'LR'})
    say('native grid ready')
    # ---- native GLM per run; run-averaged series for the responsive set
    mflat = mask.ravel(); zr, vr, Ysum = {}, {}, None
    for r in keep:
        Y4 = np.asarray(nib.load(f'{wd}/dc_run-{r}.nii.gz').dataobj, np.float32)
        par = np.loadtxt(f'{wd}/mc_run-{r}.par')
        Y = Y4.reshape(-1, Y4.shape[-1])
        z = np.zeros(Y.shape[0], np.complex128); v = np.zeros(Y.shape[0])
        z[mflat], v[mflat], _ = glm_z(Y[mflat], design(stim, par))
        zr[r], vr[r] = z, v
        Ysum = Y.astype(np.float64) if Ysum is None else Ysum + Y
    Yavg = (Ysum / len(keep)).astype(np.float32); del Ysum
    Ym = U['cat']({h: np.asarray(W_n[h] @ Yavg) for h in 'LR'})                 # run-averaged series, standard-mapped (n x T)
    Xf = design(stim, None); Xr = Xf[:, 4:]
    R2 = r2_stim(Ym.astype(np.float64), Xf, Xr); del Yavg, Ym
    halves = {'A': keep[0::2], 'B': keep[1::2]}
    zh = {X: np.mean([zr[r] for r in halves[X]], 0) for X in 'AB'}
    zn = {X: map_n(zh[X]) for X in 'AB'}
    say('native GLM done; responsive (R2 >= %.2f) vertices' % R2_THR, int((R2 >= R2_THR).sum()))
    res = dict(subject=S, runs=keep, halves=halves, acq=acq, r2_thr=R2_THR, grids={}, leak={},
               n_unknowns=int(n), n_responsive=int((R2 >= R2_THR).sum()))
    wsum = lambda m: float((area * m).sum() / max(area.sum(), 1e-30))
    res['leak']['2.0'] = dict(mean_far=wsum(far_n), gt5=wsum(far_n > 0.05), gt10=wsum(far_n > 0.10), mean_other_hemi=wsum(xh_n))
    maps = dict(idx=U['idx'].astype(np.int32), R2=R2.astype(np.float32), far_n=far_n.astype(np.float32), target_A=zn['A'].astype(np.complex64),
                target_B=zn['B'].astype(np.complex64))
    if a.gate:
        from scipy.ndimage import map_coordinates
        fimg = nib.load(f'{wd}/field_hz.nii.gz'); F = np.abs(np.asarray(fimg.dataobj, np.float64))
        mid = U['cat']({h: geo[h]['mid'] for h in 'LR'})
        q = SFM.hom(np.linalg.inv(fimg.affine) @ mimg.affine @ np.linalg.inv(A_anat), mid).T     # anatomy -> EPI voxel -> field-map voxel
        fv = map_coordinates(F, q, order=1, mode='nearest')
        respv = R2 >= R2_THR
        res['field'] = dict(p95_brain_hz=float(np.percentile(F[mask], 95)), median_responsive_hz=float(np.median(fv[respv])),
                            median_calc_responsive_hz=float(np.median(fv[respv & U['calc']])) if (respv & U['calc']).sum() else None,
                            bbr_cost=float(min(qc['bbr_costs'].values())))
        say('field statistics', {k: round(v, 3) if v is not None else None for k, v in res['field'].items()})
    if a.atlas:                                           # PLAN_ATLAS.md: x = exp(i phi) within the template's areas and the 12.4 deg stimulus radius
        At = np.load(f'{G_}/results/nyu/atlas/{S}.benson14.npz')
        assert int(At['nL']) == U['nv']['L'] and len(At['phi']) == U['nv']['L'] + U['nv']['R']
        inside = (At['varea'] > 0) & (At['eccen'] <= 12.4)
        x_atl = np.where(inside, np.exp(1j * np.radians(At['phi'].astype(np.float64))), 0)[U['idx']]
        x_atl3 = np.where((At['varea'] >= 1) & (At['varea'] <= 3) & (At['eccen'] <= 12.4), np.exp(1j * np.radians(At['phi'].astype(np.float64))), 0)[U['idx']]
        say('atlas response: vertices with a response', int((np.abs(x_atl) > 0).sum()), 'of which V1-V3', int((np.abs(x_atl3) > 0).sum()))
    sm = smoother_cache(U['Cr'], U['Gr'])
    su, fsc = sphere_frames(geo, sph, lambda V, F: vertex_areas_normals(V, F)[0])
    for g, fac in FACTORS.items():
        t1 = time.time()
        cs = coarse_shape(shape, fac); A_epi_c, A_anat_c = coarse_affine(A_epi, shape, cs), coarse_affine(A_anat, shape, cs)
        Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]
        mask_c = apply_axes(Ms, mask.astype(np.float64)) > 0.9
        zc, vc, Em = {}, {}, None
        for r in keep:
            Y4 = np.asarray(nib.load(f'{wd}/dc_run-{r}.nii.gz').dataobj, np.float32)
            Yc = apply_axes(Ms, Y4).astype(np.float32); par = np.loadtxt(f'{wd}/mc_run-{r}.par')
            Em = Yc.mean(-1) if Em is None else Em + Yc.mean(-1)
            good = Yc.mean(-1) > 0.05 * np.percentile(Yc.mean(-1), 99)
            mc_ = (mask_c & good).ravel(); Y = Yc.reshape(-1, Yc.shape[-1])
            z = np.zeros(Y.shape[0], np.complex128); v = np.zeros(Y.shape[0])
            z[mc_], v[mc_], _ = glm_z(Y[mc_], design(stim, par)); zc[r], vc[r] = z, v
            mask_c &= good
        del Y4, Yc
        tmpl = f'{tmp}/x{fac:g}_template.nii.gz'
        nib.save(nib.Nifti1Image(np.where(mask_c, Em / len(keep), 0).astype(np.float32), A_epi_c), tmpl)
        W_c, map_c = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, tmp, f'x{fac:g}')
        csp = voxel_sizes(dict(affine=A_anat_c)); tabs_c, rad_c = coarse_tabs(acq, [acq['h']] * 3, csp)
        K_c = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
        dil_c = map_c(K_c @ ones)
        Xs = {f'M_{X}': split_complex(zn[X]) for X in 'AB'}
        zcH, upH = {}, {}
        for X in 'AB':
            z_ = np.mean([zc[r] for r in halves[X]], 0); zcH[X] = np.where(mask_c.ravel(), z_, 0)
            upH[X] = map_n(upsample(zcH[X].reshape(cs), shape).ravel())
            if a.gate:
                Xs[f'P_{X}'] = split_complex(upH[X])          # CWL-P: response estimate from the coarse data alone
        if a.atlas:
            Xs['ATL'], Xs['ATL3'] = split_complex(x_atl), split_complex(x_atl3)
        C, far_c, xh_c = far_contamination_oh(W_c, K_c, U['idx'], U['nv']['L'], su, fsc, Xs)     # far leak = sf_gate.far_leak (checked: 0 difference)
        res['leak'][g] = dict(mean_far=wsum(far_c), gt5=wsum(far_c > 0.05), gt10=wsum(far_c > 0.10), mean_other_hemi=wsum(xh_c), voxel_mm=[float(x) for x in csp])
        say(g, 'mm grid', cs, 'ready', round(time.time() - t1), 's')
        maps[f'far_{g}'] = far_c.astype(np.float32)
        scores, nsets, angles, info = {}, {}, {}, {}
        for X, Yh in (('A', 'B'), ('B', 'A')):
            d = X + Yh; t2 = time.time(); rX = halves[X]
            zcX = np.mean([zc[r] for r in rX], 0); vcX = np.sum([vc[r] for r in rX], 0) / len(rX) ** 2
            zcX = np.where(mask_c.ravel(), zcX, 0); vcX = np.where(mask_c.ravel(), vcX, 0)
            target = zn[Yh]; std = map_c(zcX)
            base = (dil_c > 0.5) & (dil_n > 0.5) & (np.abs(target) > 0) & (np.abs(std) > 0)
            resp = base & (R2 >= R2_THR)
            cwlM = cwl(C[f'M_{X}'], Xs[f'M_{X}'], resp)
            clean = resp & (far_n < 0.02)
            sets = {'responsive': resp, 'clean_leak10': clean & (far_c > 0.10), 'calc': resp & U['calc']}
            dose = [('far', far_c), ('cwlM', cwlM)]
            if a.gate:
                cwlP = cwl(C[f'P_{X}'], Xs[f'P_{X}'], resp); dose.append(('cwlP', cwlP))
            if a.atlas:
                from scipy.stats import spearmanr
                cwlA, cwlA3 = cwl(C['ATL'], Xs['ATL'], resp), cwl(C['ATL3'], Xs['ATL3'], resp)
                dose += [('cwlA', cwlA), ('cwlA3', cwlA3)]
                rs = lambda u_, v_: float(spearmanr(u_[resp], v_[resp]).correlation) if resp.sum() > 30 else None
                info.setdefault(d + '_atlas', {}).update(rho_A_M=rs(cwlA, cwlM), rho_A3_M=rs(cwlA3, cwlM), rho_P_M=rs(cwlP, cwlM) if a.gate else None,
                                                         share_resp_with_atlas_response=float((np.abs(x_atl[resp]) > 0).mean()) if resp.sum() else None,
                                                         median_cwlA=float(np.median(cwlA[resp])) if resp.sum() else None)
            for nm, var in dose:
                for (lo, hi), bn in zip(BINS, BIN_NAMES):
                    sets[f'{nm}_{bn}'] = clean & (var >= lo) & (var < hi)
            preds = {'standard': std, 'native': zn[X]}
            for fw in FWHMS:
                preds[f'smooth{fw}'] = sm(fw)(std)
            up = map_n(upsample(zcX.reshape(cs), shape).ravel()); preds['upsample'] = up
            for fw in FWHMS:
                preds[f'upsmooth{fw}'] = sm(fw)(up)
            # model: white-noise likelihood, Matern prior, settings of this grid
            std_sd = np.sqrt(U['cat']({h: W_c[h].multiply(W_c[h]) @ vcX for h in 'LR'}))
            amp = np.abs(std); rr = (dil_c > 0.5) & (amp > 3 * np.maximum(std_sd, 1e-12))
            sig0 = float(np.median(amp[rr] / np.maximum(dil_c[rr], 0.3))) if rr.sum() > 50 else float(np.median(amp[base]))
            prob = Problem(K_c, mask_c.ravel(), vcX, U['Cr'], U['Gr'], U['nroi'])
            B = prob.rhs(zcX); X0 = None; its = {}
            for rho, ss in grid_settings[g]:
                prob.set_prior(rho, ss * sig0); Xh, it, _ = prob.solve(B, tol=1e-4, maxit=500, X0=X0); X0 = Xh
                f = Xh[:, 0] + 1j * Xh[:, 1]
                preds[f'model_r{rho:g}_s{ss:g}'] = map_n(np.asarray(K_n @ f.real.astype(np.float32)) + 1j * np.asarray(K_n @ f.imag.astype(np.float32)))
                its[f'r{rho:g}_s{ss:g}'] = int(it)
            del prob
            if a.gate:                                         # PLAN_NYU_GATE A: model where CWL-P > tau, competitor elsewhere
                mk = [k for k in preds if k.startswith('model_')][0]
                for tau in (0.0, 0.02, 0.05, 0.10, 0.20):
                    preds[f'gate_t{tau:g}'] = np.where(cwlP > tau, preds[mk], preds[a.comp])
                    info.setdefault(d + '_gate_share', {})[f't{tau:g}'] = float((cwlP[resp] > tau).mean()) if resp.sum() else None
            scores[d] = {m: {s: (resid_c(target[sel], p[sel]) if sel.sum() >= MIN_N else None) for s, sel in sets.items()} for m, p in preds.items()}
            angles[d] = {m: angle_err(target[resp], p[resp]) for m, p in preds.items() if resp.sum() >= MIN_N}
            nsets[d] = {s: int(sel.sum()) for s, sel in sets.items()}
            info[d] = dict(sigma0=sig0, iterations=its, t_s=round(time.time() - t2), median_cwlM_responsive=float(np.median(cwlM[resp])) if resp.sum() else None)
            if a.save_maps:
                maps[f'cwlM_{g}_{d}'] = cwlM.astype(np.float32)
                if a.gate:
                    maps[f'cwlP_{g}_{d}'] = cwlP.astype(np.float32)
                for m in ['standard', 'upsample'] + [k for k in preds if k.startswith('upsmooth') or k.startswith('smooth') or k.startswith('model_')]:
                    maps[f'{m}_{g}_{d}'] = preds[m].astype(np.complex64)
            say(g, X, '->', Yh, 'responsive', int(resp.sum()), 'quick:', {m: round(scores[d][m]['responsive'], 4) for m in ('standard', 'upsample', 'upsmooth6', 'smooth8') if scores[d][m]['responsive'] is not None},
                {m: round(v['responsive'], 4) for m, v in scores[d].items() if m.startswith('model_') and v['responsive'] is not None}, round(time.time() - t2), 's')
        avg = {}
        for m in scores['AB']:
            avg[m] = {}
            for s in scores['AB'][m]:
                v = [scores[d][m][s] for d in scores if scores[d][m][s] is not None]
                avg[m][s] = float(np.mean(v)) if len(v) == len(scores) else None
        res['grids'][g] = dict(scores=avg, by_direction=scores, angle_deg={m: float(np.mean([angles[d][m] for d in angles if m in angles[d]])) for m in angles.get('AB', {})},
                               n=nsets, info=info, voxel_mm=[float(x) for x in csp])
        json.dump(res, open(f'{OUT}/{S}.nyu_scores.json', 'w'), indent=1, default=float)
    if a.save_maps:
        np.savez_compressed(f'{OUT}/{S}.nyu_maps.npz', **maps)
    res['t_total_s'] = round(time.time() - T0)
    json.dump(res, open(f'{OUT}/{S}.nyu_scores.json', 'w'), indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    say('done')


if __name__ == '__main__':
    main()
