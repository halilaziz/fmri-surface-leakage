"""Week 7, ds002702 (Kay et al.): real second acquisition in the same people. A source session (3T 2.4 mm GE, or 7T 1.05 mm SE)
predicts the 7T 0.8 mm GE session's standard ribbon-mapped maps of the six ring conditions (all runs). Each session is modelled on
its own native grid, registered separately to the authors' FreeSurfer anatomy (kay_prep.py; no distortion correction in stage 1).
Every setting is scored and saved; the pre-specified choice (kay_prespec.py, from the within-session emulation of all five
subjects at the factor matching the source voxel) is applied by summarize_kay_cross.py:
  smooth{fw}            the source's own standard map W_s y_s, surface smoothing (fw 0 = standard)
  resample / resmooth   source betas resampled onto the target grid (cubic B-spline; the cross-session analogue of band-limited
                        upsampling), the target's weights W_t, smoothing
  model                 posterior mean from the source (source PSF; white or correlated noise; the emulation's prior grid),
                        rendered W_t K_t f through the target's operator
Scored on vertices covered by both sessions (and visual labels); residual energy fraction per ring map after its best scaling;
mixing bins by the far leak of standard mapping at the source grid (vertices clean at the target grid). Reference: the target's
own odd / even split.
Usage (xcp_d container): python3 kay_cross.py SUBJECT SOURCE_SESSION [--target ring7T] [--dc]
"""
import os, sys, json, time, argparse, shutil, glob
import numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kay_emul as KE
from kay_emul import *

OUTX = f'{KE.G_}/results/kay/cross'
QC_MIN_R = 0.1           # runs whose betas correlate < 0.1 with the mean of the session's other runs are dropped (kay_betaqc.py on
                         # the uncorrected prep, applied to both stages; fixed before any cross-session result was computed)
T2_SE_7T = 50.0          # spin echo: decay across the readout ~ grey-matter T2 at 7T (the T2' part refocuses at the echo)


def session(S, ses, prep):
    wd = f'{prep}/{S}/{ses}'
    mimg = nib.load(f'{wd}/session_mean.nii.gz')
    js = json.load(open(sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/func/*task-ring_run-01_bold.json'))[0]))
    acq = acq_from_json(js, '3T' if js['MagneticFieldStrength'] < 5 else '7T')
    se = 'ep2d_se' in js.get('PulseSequenceDetails', '') or js.get('SequenceName', '').startswith('epse')
    if se:
        acq['t2s_ms'] = T2_SE_7T
    Y = np.stack([np.asarray(nib.load(f'{wd}/runbeta_{c}.nii.gz').dataobj, np.float32) for c in CONDS], 3)   # x y z cond run
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    qc = json.load(open(f'{wd}/{S}_{ses}_prep_qc.json'))
    rq = json.load(open(f'{PREPD}/beta_qc.json'))[f'{S}/{ses}']['r_loo']
    keep = [i for i, r in enumerate(rq) if r >= QC_MIN_R]
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    return dict(ses=ses, wd=wd, img=mimg, A_epi=mimg.affine, shape=tuple(mimg.shape[:3]), A_anat=np.loadtxt(f'{wd}/epi2anat_vox2ras.txt'),
                acq=acq, spin_echo=se, Y=Y[..., keep] * mask[..., None, None], mask=mask, qc=qc, runs_kept=[i + 1 for i in keep],
                runs_dropped=[i + 1 for i in range(len(rq)) if i not in keep], mrun=mrun[..., keep])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('source'); ap.add_argument('--target', default='ring7T')
    ap.add_argument('--dc', action='store_true', help='field-map corrected sessions (kay_undistort.py, results/kay/prep_dc)')
    a = ap.parse_args(); S = a.subject
    tagf = '.dc' if a.dc else ''
    # stage 2: each session's field-map correction is used only where it improved the registration QC (fraction of vertices with
    # grey > white EPI contrast across the white surface; kay_stage2_select.py, written before any cross-session result)
    sel = json.load(open(f'{KE.G_}/results/kay/stage2_selection.json')) if a.dc else {}
    prep_of = lambda ses: f"{KE.G_}/results/kay/{sel[f'{S}/{ses}']['use']}" if a.dc else PREPD
    od = f'{OUTX}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{a.source}{tagf}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, a.source, '->', a.target, f'{time.time() - T0:7.0f}s', *x, flush=True)
    geo, sph = load_anatomy_fs(f"{FSD}/subj{S.split('-')[1]}")
    s, t = session(S, a.source, prep_of(a.source)), session(S, a.target, prep_of(a.target))
    U = unknowns(geo, [(s['A_anat'], s['shape']), (t['A_anat'], t['shape'])])
    n = U['n']; ones = np.ones(n, np.float32)
    say('unknowns', n, 'source acq', {k: round(v, 3) if isinstance(v, float) else v for k, v in s['acq'].items()})
    for x, tag in ((s, 'src'), (t, 'tgt')):
        tabs, rad = native_tabs(x['acq'])
        x['K'] = build_operator(U['P'], U['V'], U['Cc'], n, dict(shape=x['shape'], affine=x['A_anat']), tabs, rad).tocsr()
        x['W'], x['map'] = ribbon_weights(geo, U, x['A_epi'], x['A_anat'], f"{x['wd']}/session_mean.nii.gz", tmp, f"{x['ses']}_{tag}")
        x['dil'] = x['map'](x['K'] @ ones)
        x['far'], x['xh'] = far_leak(x['W'], x['K'], U, geo, sph)
        R = x['Y'].shape[-1]; x['R'] = R
        x['z'] = x['map'](x['Y'].mean(-1).reshape(-1, len(CONDS)))
        x['cov'] = (x['dil'] > 0.5) & (np.abs(x['z']).sum(1) > 0)
        say(x['ses'], 'grid', x['shape'], np.round(voxel_sizes(dict(affine=x['A_anat'])), 2), 'runs', R, 'covered', int(x['cov'].sum()),
            'median far leak (visual)', round(float(np.median(x['far'][U['visual'] & x['cov']])), 3))
    zt_A = t['map'](t['Y'][..., 0::2].mean(-1).reshape(-1, len(CONDS))); zt_B = t['map'](t['Y'][..., 1::2].mean(-1).reshape(-1, len(CONDS)))
    base = s['cov'] & t['cov']
    npz = dict(idx=U['idx'].astype(np.int32), visual=U['visual'], calc=U['calc'], base=base, target=t['z'].astype(np.float32),
               target_A=zt_A.astype(np.float32), target_B=zt_B.astype(np.float32), source_native=s['z'].astype(np.float32),
               far_s=s['far'].astype(np.float32), far_t=t['far'].astype(np.float32), xh_s=s['xh'].astype(np.float32),
               xh_t=t['xh'].astype(np.float32))
    sm = smoother_cache(U['Cr'], U['Gr'])
    smm = lambda fw, X: X if fw == 0 else np.stack([sm(fw)(X[:, j]) for j in range(X.shape[1])], 1)
    # ---- competitors
    for fw in (0.0, 2.0, 4.0, 6.0, 8.0):
        npz[f'smooth{fw:g}'] = smm(fw, s['z']).astype(np.float32)
    ijk = np.stack(np.meshgrid(*[np.arange(d) for d in t['shape']], indexing='ij'), -1).reshape(-1, 3).astype(np.float64)
    q = SFM.hom(np.linalg.inv(s['A_anat']) @ t['A_anat'], ijk).T
    ys = s['Y'].mean(-1)
    for order, lab in ((3, 'resample'), (1, 'reslin')):
        rs = np.stack([map_coordinates(ys[..., j], q, order=order, cval=0.0, prefilter=True) for j in range(len(CONDS))], 1)
        zr = t['map'](rs)
        for fw in ((0.0, 2.0, 4.0) if order == 3 else (0.0,)):
            npz[lab if fw == 0 else f'{lab}{fw:g}'] = smm(fw, zr).astype(np.float32)
    del q, ijk
    say('competitors ready')
    # ---- model: noise from the source's run-to-run variance (native grid; no emulation, no calibration factor)
    Ys = s['Y']; R = s['R']; vox_s = voxel_sizes(dict(affine=s['A_anat']))
    s2 = np.where(s['mask'], Ys.var(-1, ddof=1).mean(3), 0).ravel() / R
    res_maps = []
    for j in range(len(CONDS)):
        Yj = Ys[..., j, :]; r_ = Yj - Yj.mean(-1, keepdims=True); sd = np.sqrt((r_ ** 2).sum(-1) / (R - 1))
        ok = s['mask'] & (sd > 0)
        res_maps += [np.where(ok, r_[..., i] / np.where(ok, sd, 1.0), 0.0) for i in range(R)]
    Pk, kc, prof = noise_spectrum(res_maps, s['mask'], vox_s, s['shape']); del res_maps
    std_sd = np.sqrt(U['cat']({h: s['W'][h].multiply(s['W'][h]) @ s2 for h in 'LR'}))
    amp = np.sqrt((s['z'] ** 2).mean(1)); resp = s['cov'] & (amp > 3 * np.maximum(std_sd, 1e-12))
    sig0 = float(np.median(amp[resp] / np.maximum(s['dil'][resp], 0.3))) if resp.sum() > 50 else float(np.median(amp[s['cov']]))
    ys_flat = Ys.mean(-1).reshape(-1, len(CONDS))
    fac = float(np.mean(voxel_sizes(dict(affine=s['A_anat']))) / np.mean(voxel_sizes(dict(affine=t['A_anat']))))
    MODELS = models_for(fac)                       # the emulation's grid for this voxel ratio (x3 -> coarse, x1.3 -> fine)
    its = {}
    for lab in ('white', 'corr'):
        ms = [(r_, s_) for l_, r_, s_ in MODELS if l_ == lab]
        prob = (Problem(s['K'], s['mask'].ravel(), s2, U['Cr'], U['Gr'], U['nroi']) if lab == 'white'
                else ProblemCorr(s['K'], s['mask'].ravel(), s2, U['Cr'], U['Gr'], U['nroi'], s['shape'], Pk))
        if not hasattr(prob, 'sqD'):
            prob.sqD = np.sqrt(prob.D).astype(np.float64)
        B = rhs_multi(prob, ys_flat); X0 = None
        for rho, gs in ms:
            t1 = time.time(); prob.set_prior(rho, gs * sig0); Xh, it, _ = prob.solve(B, tol=1e-4, maxit=500, X0=X0); X0 = Xh
            npz[f'{lab}_r{rho:g}_s{gs:g}'] = t['map'](np.asarray(t['K'] @ Xh.astype(np.float32))).astype(np.float32)
            its[f'{lab}_r{rho:g}_s{gs:g}'] = dict(iterations=int(it), t_s=round(time.time() - t1))
        del prob
        say(lab, 'models done', {k: v['iterations'] for k, v in its.items() if k.startswith(lab)})
    # ---- quick scores (visual, covered by both) for the log; the pre-specified comparison is made in the summary
    sel = base & U['visual']
    keys = [k for k in npz if k.startswith(('smooth', 'resample', 'reslin', 'white_', 'corr_'))]
    quick = {k: round(resid_multi(npz['target'], npz[k], sel), 4) for k in keys}
    ceil = 0.5 * (resid_multi(zt_B, zt_A, sel) + resid_multi(zt_A, zt_B, sel))
    np.savez_compressed(f'{od}/{S}.{a.source}-{a.target}{tagf}.kay_cross.npz', **npz)
    log = dict(subject=S, source=a.source, target=a.target, field_map_corrected=bool(a.dc), source_prep=s['wd'], target_prep=t['wd'], source_acq=s['acq'], target_acq=t['acq'], source_spin_echo=s['spin_echo'],
               source_runs=s['R'], target_runs=t['R'], source_runs_dropped=s['runs_dropped'], target_runs_dropped=t['runs_dropped'], source_bbr=s['qc'].get('bbr_costs'), target_bbr=t['qc'].get('bbr_costs'),
               sigma0=sig0, n_responsive=int(resp.sum()), voxel_ratio=fac, models=[list(m) for m in MODELS], iterations=its, n_base=int(base.sum()), n_visual=int(sel.sum()),
               noise_profile=dict(k_cyc_per_mm=kc.tolist(), power_rel_white=prof.tolist()), visual_quick=quick,
               target_split_half_visual=ceil, t_s=round(time.time() - T0))
    json.dump(log, open(f'{od}/{S}.{a.source}-{a.target}{tagf}.kay_cross_log.json', 'w'), indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    say('done; n visual', int(sel.sum()))


if __name__ == '__main__':
    main()
