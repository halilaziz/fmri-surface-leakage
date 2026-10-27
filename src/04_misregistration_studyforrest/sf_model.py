"""Week 4, StudyForrest: real multi-resolution validation for one subject (xcp_d container: numpy/scipy/nibabel + wb_command).

Every session (7T 0.8 / 1.4 / 2.0 / 3.0 mm; 3T 1.4 / 2.0 / 3.0 mm where available) is modelled on its NATIVE acquisition grid,
expressed in the subject's anatomical space through its bbregister registration. For each ordered pair (source s coarser than
target t, or 3T source -> 7T target) the source data predict the target session's standard ribbon-mapped map of the left- and
right-hemifield betas:
  standard            W_s y_s                               (the source's own surface map)
  smoothing           surface smoothing of W_s y_s (10 mm; FWHM fixed from the HCP tuning)
  resample_smoothing  y_s resampled onto the target grid (trilinear), W_t, smoothing (8 mm; HCP)
  model               posterior mean f from the source (operator with the source's acquisition PSF; prior fixed from the HCP
                      tuning by source resolution), rendered W_t K_t f
Scored on vertices covered well by both sessions (and on visual-cortex labels), residual energy fraction after the best real
scaling per hemifield map. Reference points: the target's odd-run map predicting its even-run map (and vice versa).
Usage (xcp_d container): python3 sf_model.py SUBJECT [--threads 8]
"""
import os, sys, json, glob, time, argparse
import numpy as np, scipy.sparse as sp, nibabel as nib
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
from w3lib import *

ROOT = os.environ.get('GEO_ROOT', '/path/to/geo_root')
SF = f'{ROOT}/data/studyforrest'
PREP = f'{ROOT}/results/week4/sf/prep'
OUT = f'{ROOT}/results/week4/sf/model'
# 7T sessions (Sengupta et al. 2017, Data in Brief Table 2): voxel, PE lines, FOV along PE; the echo spacing is not reported and is
# assumed (unaccelerated 1.0 / 0.8 / 0.7 / 0.6 ms); acceleration and PE axis come from each session's sidecar
ACQ7 = {'r08': dict(h=0.8, pe_lines=208, fov_pe=166.4, esp_ms=1.0), 'r14': dict(h=1.4, pe_lines=140, fov_pe=196.0, esp_ms=0.8),
        'r20': dict(h=2.0, pe_lines=100, fov_pe=200.0, esp_ms=0.7), 'r30': dict(h=3.0, pe_lines=66, fov_pe=198.0, esp_ms=0.6)}
T2S = {'7T': 28.0, '3T': 45.0}
PRIOR = {'r14': (12.0, 0.125), 'r20': (12.0, 0.125), 'r30': (24.0, 0.25)}     # HCP week-3 tuning (2.4 mm -> 12/0.125, 3.2 mm -> 24/0.25)
FW_SMOOTH, FW_RESAMPLE = 10.0, 8.0                                            # HCP week-3 tuning of the competitors
X_FW, X_FW_MODEL = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0), (2.0, 4.0, 6.0, 8.0)   # exploratory FWHM curves (logged as x_*)
PRIOR_GRID = [(r, g) for r in (3.0, 6.0, 12.0, 24.0) for g in (0.125, 0.25, 0.5, 1.0, 2.0)]  # exploratory prior grid (--prior-grid)
VISUAL = (b'pericalcarine', b'cuneus', b'lingual', b'lateraloccipital')
RES_MM = {'r08': 0.8, 'r14': 1.4, 'r20': 2.0, 'r30': 3.0}


def read_lta(p):
    L = open(p).read().split('\n')
    typ = int([l for l in L if l.strip().startswith('type')][0].split('=')[1].split()[0])
    i = [k for k, l in enumerate(L) if l.strip() == '1 4 4'][0]
    return typ, np.array([[float(x) for x in L[i + 1 + r].split()] for r in range(4)])


def write_surf(path, V, F, hemi):
    da = [nib.gifti.GiftiDataArray(V.astype(np.float32), intent='NIFTI_INTENT_POINTSET', datatype='NIFTI_TYPE_FLOAT32'),
          nib.gifti.GiftiDataArray(F.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE', datatype='NIFTI_TYPE_INT32')]
    da[0].meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': {'L': 'CortexLeft', 'R': 'CortexRight'}[hemi]})
    nib.save(nib.gifti.GiftiImage(darrays=da), path)


def hom(A, X):
    return X @ A[:3, :3].T + A[:3, 3]


def acq_params(field, res, S):
    if field == '7T':
        js = json.load(open(glob.glob(f'{SF}/multires7t/{S}/ses-{res}/{S}_ses-{res}_task-orientation_rec-dico_bold.json')[0]))
        p = dict(ACQ7[res]); R = js.get('ParallelReductionFactorInPlane', 1)
        pe_axis = {'i': 0, 'j': 1}[js['PhaseEncodingDirection'][0]]
        esp_eff = p['esp_ms'] / R; h = p['h']; fov = p['fov_pe']
    else:
        js = json.load(open(sorted(glob.glob(f'{SF}/multires3t/{S}/func/{S}_task-orientation_acq-{res}_run-01_bold.json'))[0]))
        pe_axis = {'i': 0, 'j': 1}[js['PhaseEncodingDirection'][0]]
        esp_eff = 1000 * js['EffectiveEchoSpacing']; h = float(js['SliceThickness'])
        fov = js['AcquisitionMatrixPE'] * h
    return dict(h=h, pe_axis=pe_axis, esp_eff_ms=esp_eff, fov_pe_mm=fov, t2s_ms=T2S[field], slice_edge=0.15, taper=0.3,
                sigma_interp=0.3 * h)


def session_tabs(acq):
    """Native-acquisition kernels along the three EPI voxel axes (unit integral), support (4, 4, 3) voxels as in week 2."""
    rad = SUPPORT_VOX['1.6']
    spacing = [acq['h'], acq['h'], 1.1 * acq['h']]                   # 10% slice gap
    tabs = [kernel_table(d, 1.0, acq=acq, trunc_mm=(rad[d] + 0.5) * spacing[d], R_mm=(rad[d] + 1) * spacing[d]) for d in range(3)]
    return tabs, rad


def resid_real(target, pred, sel):
    """Residual energy fraction after the best real scaling, pooled over the two hemifield maps (real, imaginary parts)."""
    num = den = 0.0
    for part in (np.real, np.imag):
        z, p = part(target[sel]), part(pred[sel])
        c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--prior-grid', action='store_true', help='also score the model over PRIOR_GRID (exploratory, x_model_r*_s*)')
    a = ap.parse_args(); S = a.subject
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    T0 = time.time(); log = {'subject': S, 'sessions': {}, 'pairs': {}}
    say = lambda *x: print(S, f'{time.time() - T0:7.0f}s', *x, flush=True)

    # ---- anatomy (FreeSurfer, scanner RAS of orig.mgz)
    FS = f'{SF}/freesurfer/{S}'
    orig = nib.load(f'{FS}/mri/orig.mgz'); v2r = orig.affine; tkr2scan = v2r @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    geo = {}
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        W, F = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.white'); Pv, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.pial')
        sph, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.sphere')
        lab, _, names = nib.freesurfer.read_annot(f'{FS}/label/{fh}.aparc.annot')
        nm = np.array([names[l] if l >= 0 else b'unknown' for l in lab])
        W, Pv = hom(tkr2scan, W), hom(tkr2scan, Pv)
        geo[h] = dict(white=W, pial=Pv, mid=0.5 * (W + Pv), faces=F.astype(np.int64), sphere=sph,
                      roi=~np.isin(nm, [b'unknown', b'corpuscallosum']), visual=np.isin(nm, VISUAL), calc=nm == b'pericalcarine')
    nv = {h: len(geo[h]['white']) for h in 'LR'}

    # ---- sessions: native grids in anatomical space, data, registration QC
    ses = {}
    for key in sorted(os.listdir(f'{PREP}/{S}')):
        wd = f'{PREP}/{S}/{key}'
        if not os.path.exists(f'{wd}/betaR_even.nii.gz'):
            continue
        field, res = key.split('-')
        mimg = nib.load(f'{wd}/session_mean.nii.gz')
        typ, M = read_lta(f'{wd}/bbr.lta')
        A_anat = v2r @ M if typ == 0 else M @ mimg.affine              # EPI voxel -> anatomical scanner RAS
        vol = lambda n: np.asarray(nib.load(f'{wd}/{n}.nii.gz').dataobj, np.float64).ravel()
        ses[key] = dict(field=field, res=res, shape=mimg.shape[:3], A_anat=A_anat, A_epi=mimg.affine, wd=wd,
                        y=vol('betaL') + 1j * vol('betaR'), y_odd=vol('betaL_odd') + 1j * vol('betaR_odd'),
                        y_even=vol('betaL_even') + 1j * vol('betaR_even'), s2=vol('varL') + vol('varR'),
                        mask=vol('mask') > 0.5, acq=acq_params(field, res, S))
        qc = json.load(open(f'{PREP}/{S}/{S}.sf_prep_qc.json'))['sessions'].get(key, {})
        rr = f'{PREP}/{S}/{S}.sf_rereg_qc.json'
        rq = json.load(open(rr)).get(key, {}) if os.path.exists(rr) else {}
        log['sessions'][key] = dict(bbr_cost=rq.get('cost', qc.get('bbr_cost')), bbr_start=rq.get('best', 'flirt'), acq=ses[key]['acq'])
    say('sessions:', list(ses))

    # ---- unknowns: cortex whose midthickness lies inside any session's field of view (5 mm margin)
    inside = {h: np.zeros(nv[h], bool) for h in 'LR'}
    for s in ses.values():
        Ai = np.linalg.inv(s['A_anat']); vs = voxel_sizes(dict(affine=s['A_anat'])); mg = 5.0 / vs
        for h in 'LR':
            q = hom(Ai, geo[h]['mid'])
            inside[h] |= ((q > -0.5 - mg) & (q < np.array(s['shape']) - 0.5 + mg)).all(1)
    for h in 'LR':
        geo[h]['roi'] = geo[h]['roi'] & inside[h]
    av = analysis_vertices(geo)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])]); n = len(idx)
    nroi = {h: int(av[h].sum()) for h in 'LR'}
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(n)
    Cr, Gr = {}, {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(geo[h]['mid'], geo[h]['faces'], av[h]); Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    P, V, C, Z, _ = subject_points(geo)
    keep = loc[C] >= 0; P, V, Cc = P[keep], V[keep], loc[C[keep]]
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    visual = cat({h: geo[h]['visual'] for h in 'LR'}); calc = cat({h: geo[h]['calc'] for h in 'LR'})
    say('unknowns', n, nroi)

    # ---- per session: Workbench ribbon weights on the native grid (surfaces moved into the EPI's world space), operator
    for key, s in ses.items():
        t0 = time.time()
        a2e = s['A_epi'] @ np.linalg.inv(s['A_anat'])                   # anatomical RAS -> EPI world coordinates
        W = {}
        for h in 'LR':
            for nm in ('white', 'pial', 'mid'):
                write_surf(f'{od}/{key}.{h}.{nm}.surf.gii', hom(a2e, geo[h][nm]), geo[h]['faces'], h)
            txt = f'{od}/{key}.{h}.w.txt'
            wb_ribbon_weights(f"{s['wd']}/session_mean.nii.gz", f'{od}/{key}.{h}.mid.surf.gii', f'{od}/{key}.{h}.white.surf.gii',
                              f'{od}/{key}.{h}.pial.surf.gii', txt)
            W[h] = parse_weights_text(txt, nv[h], s['shape']).tocsr(); os.remove(txt)
            for nm in ('white', 'pial', 'mid'):
                os.remove(f'{od}/{key}.{h}.{nm}.surf.gii')
        s['W'] = W
        s['map'] = lambda y, W=W: cat({h: W[h] @ y for h in 'LR'})
        tabs, rad = session_tabs(s['acq'])
        s['K'] = build_operator(P, V, Cc, n, dict(shape=s['shape'], affine=s['A_anat']), tabs, rad).tocsr()
        s['z'] = s['map'](s['y']); s['z_odd'] = s['map'](s['y_odd']); s['z_even'] = s['map'](s['y_even'])
        s['dil'] = s['map'](s['K'] @ np.ones(n, np.float32))                       # partial-volume dilution of standard mapping
        s['std_sd'] = np.sqrt(cat({h: W[h].multiply(W[h]) @ np.where(s['mask'], s['s2'], 0) for h in 'LR'}))
        s['covered'] = (s['dil'] > 0.5) & (np.abs(s['z']) > 0)
        log['sessions'][key].update(nnz_K=int(s['K'].nnz), covered_vertices=int(s['covered'].sum()), t_setup_s=round(time.time() - t0))
        say(key, 'weights + operator', round(time.time() - t0), 's; covered', int(s['covered'].sum()))

    # ---- sources: model estimates; then every target finer than the source (or any 7T target for a 3T source)
    order = lambda k: (RES_MM[ses[k]['res']], ses[k]['field'])
    sm_cache = {}
    smooth = lambda fw: sm_cache.setdefault(fw, surface_smooth(Cr, Gr, fw))
    preds = {}
    for skey in sorted(ses, key=order, reverse=True):
        s = ses[skey]
        if s['res'] == 'r08':
            continue
        rho, ss = PRIOR[s['res']]
        t0 = time.time()
        prob = Problem(s['K'], s['mask'], s['s2'], Cr, Gr, nroi)
        resp = s['covered'] & (np.abs(s['z']) > 3 * np.maximum(s['std_sd'], 1e-12))       # responsive, judged on the source only
        sig0 = float(np.median(np.abs(s['z'][resp]) / np.maximum(s['dil'][resp], 0.3))) if resp.sum() > 50 else float(np.median(np.abs(s['z'][s['covered']])))
        prob.set_prior(rho, ss * sig0)
        X, it, _ = prob.solve(prob.rhs(s['y']), tol=1e-4, maxit=500)
        f = X[:, 0] + 1j * X[:, 1]
        log['sessions'][skey].update(model_iterations=it, sigma0=sig0, n_responsive=int(resp.sum()), t_model_s=round(time.time() - t0))
        say(skey, 'model estimate', it, 'iterations,', round(time.time() - t0), 's')
        fx = {}
        if a.prior_grid:
            B = prob.rhs(s['y']); Xg = None; gi = {}
            for rg, sg in PRIOR_GRID:
                t1 = time.time(); prob.set_prior(rg, sg * sig0)
                Xg, itg, relg = prob.solve(B, tol=1e-4, maxit=400, X0=Xg)
                fx[f'x_model_r{rg:g}_s{sg:g}'] = Xg[:, 0] + 1j * Xg[:, 1]
                gi[f'r{rg:g}_s{sg:g}'] = dict(iterations=int(itg), max_rel_residual=float(np.max(relg)), t_s=round(time.time() - t1))
            log['sessions'][skey]['prior_grid'] = gi
            say(skey, 'prior grid', {k: v['iterations'] for k, v in gi.items()}, round(time.time() - t0), 's')
        for tkey in sorted(ses, key=order):
            t = ses[tkey]
            finer = RES_MM[t['res']] < RES_MM[s['res']] or (s['field'] == '3T' and t['field'] == '7T')
            if tkey == skey or not finer or t['field'] != '7T':
                continue
            # resample the source volume onto the target grid (trilinear), then map with the target's weights
            ijk_t = np.stack(np.meshgrid(*[np.arange(d) for d in t['shape']], indexing='ij'), -1).reshape(-1, 3)
            q = hom(np.linalg.inv(s['A_anat']) @ t['A_anat'], ijk_t.astype(np.float64)).T
            ys = s['y'].reshape(s['shape'])
            res_vol = map_coordinates(ys.real, q, order=1, cval=0) + 1j * map_coordinates(ys.imag, q, order=1, cval=0)
            rendered = (t['K'] @ f.real.astype(np.float32)) + 1j * (t['K'] @ f.imag.astype(np.float32))   # K_t f on the target grid
            P_ = {'standard': s['z'], 'smoothing': smooth(FW_SMOOTH)(s['z']),
                  'resample_smoothing': smooth(FW_RESAMPLE)(t['map'](res_vol)), 'model': t['map'](rendered)}
            # exploratory (not pre-specified): smoothing-FWHM curves of the competitors, and the model prediction post-smoothed
            resampled = t['map'](res_vol)
            P_.update({f'x_smooth{fw:g}': smooth(fw)(s['z']) for fw in X_FW if fw != FW_SMOOTH})
            P_.update({f'x_resample_smooth{fw:g}': smooth(fw)(resampled) for fw in X_FW if fw != FW_RESAMPLE})
            P_.update({f'x_model_smooth{fw:g}': smooth(fw)(P_['model']) for fw in X_FW_MODEL})
            for mname, fg in fx.items():
                P_[mname] = t['map']((t['K'] @ fg.real.astype(np.float32)) + 1j * (t['K'] @ fg.imag.astype(np.float32)))
            sel_all = s['covered'] & t['covered']
            sets = {'covered': sel_all, 'visual': sel_all & visual, 'pericalcarine': sel_all & calc}
            sc = {'n': {k: int(v.sum()) for k, v in sets.items()}}
            for mname, p in P_.items():
                sc[mname] = {k: (resid_real(t['z'], p, v) if v.sum() >= 50 else None) for k, v in sets.items()}
            sc['ceiling_split_half'] = {k: (0.5 * (resid_real(t['z_even'], t['z_odd'], v) + resid_real(t['z_odd'], t['z_even'], v))
                                            if v.sum() >= 50 else None) for k, v in sets.items()}
            log['pairs'][f'{skey}->{tkey}'] = sc
            preds[f'{skey}->{tkey}'] = {m: p.astype(np.complex64) for m, p in P_.items() if not m.startswith('x_')}
            say(f'{skey} -> {tkey}', {m: round(v['covered'], 3) for m, v in sc.items() if m not in ('n',) and not m.startswith('x_')
                                      and v.get('covered') is not None})
        del prob
    np.savez_compressed(f'{od}/{S}.sf_predictions.npz', idx=idx.astype(np.int32), visual=visual, calc=calc,
                        **{f'z_{k}': s['z'].astype(np.complex64) for k, s in ses.items()},
                        **{f'zodd_{k}': s['z_odd'].astype(np.complex64) for k, s in ses.items()},
                        **{f'zeven_{k}': s['z_even'].astype(np.complex64) for k, s in ses.items()},
                        **{f'covered_{k}': s['covered'] for k, s in ses.items()},
                        **{f'{pk}|{m}': v for pk, d in preds.items() for m, v in d.items()})
    log['t_total_s'] = round(time.time() - T0)
    json.dump(log, open(f'{od}/{S}.sf_model_log.json', 'w'), indent=1, default=float)
    say('done')


if __name__ == '__main__':
    main()
