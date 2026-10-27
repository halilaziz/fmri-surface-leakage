"""Week 7, ds002702 (Kay et al., 7T 0.8 mm / 7T SE 1.05 mm / 3T 2.4 mm eccentricity rings): preprocessing of one subject session on
its native grid (fmriprep container: FSL + python), as week 4's sf_prep.py:
  1. motion correction of every run to the temporal mean of run 1 (mcflirt, spline);
  2. registration of the session mean to the FreeSurfer anatomy of the authors (OSF j2wsc): FLIRT (6 dof, correlation ratio) from
     several starts, then FSL boundary-based registration (flirt -cost bbr) with the FreeSurfer cerebral white matter as wmseg; the
     start with the lowest BBR cost is kept; saved as the 4x4 EPI voxel -> anatomical scanner RAS matrix;
  3. GLM per run: six ring conditions (3.5 s; SPM double-gamma HRF + temporal derivative), Legendre drifts (order 3), 6 motion
     parameters; betas in percent signal change; per-run betas kept (for within-session splits).
No distortion correction (stage 1; source and target of the within-session test share the session's distortion).
Usage (fmriprep container): python3 kay_prep.py SUBJECT SESSION     e.g. sub-01 ring7T
"""
import os, sys, json, glob, time, subprocess
import numpy as np, nibabel as nib
from scipy.stats import gamma
from concurrent.futures import ThreadPoolExecutor

G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
DATA = f'{G}/data/ds002702'
FSD = f'{G}/data/ds002702_freesurfer'
OUT = f'{G}/results/kay/prep'
CONDS = [f'ecc{k}' for k in range(1, 7)]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'failed ({r.returncode}): {cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r.stdout


def hrf(tr, n=32):
    dt = tr / 16; t = np.arange(0, n, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return h / h.sum(), dt


def design(ev, nvol, tr):
    h, dt = hrf(tr); tt = np.arange(0, nvol * tr, dt); cols = []
    idx = (np.arange(nvol) * tr / dt + 0.5 * tr / dt).astype(int)
    for c in CONDS:
        box = np.zeros(len(tt))
        for on, du, ty in ev:
            if ty == c:
                box[(tt >= on) & (tt < on + du)] = 1.0
        x = np.convolve(box, h)[:len(tt)]
        cols += [x[idx], np.gradient(x)[idx]]
    return np.stack(cols, 1)                                  # columns: ecc1, d_ecc1, ecc2, d_ecc2, ...


def read_events(p):
    L = open(p).read().strip().split('\n')[1:]
    return [(float(a.split('\t')[0]), float(a.split('\t')[1]), a.split('\t')[2].strip()) for a in L]


def glm_run(bold, par, ev, tr, mask):
    Y = np.asarray(nib.load(bold).dataobj, np.float32)[mask]
    nvol = Y.shape[1]
    x = np.linspace(-1, 1, nvol)
    leg = np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(4)], 1)
    X = np.concatenate([design(ev, nvol, tr), leg, np.loadtxt(par)[:nvol]], 1)
    B, *_ = np.linalg.lstsq(X, Y.T.astype(np.float64), rcond=None)
    mean = np.maximum(Y.mean(1), 1e-6)
    return np.stack([100 * B[2 * k] / mean for k in range(len(CONDS))], 0)      # (6, nmask) percent signal change


def fsl_scaled(img):
    """FSL's 'scaled voxel' coordinates: voxel * pixdim, with x flipped when the affine has a positive determinant."""
    n = img.shape; z = img.header.get_zooms()[:3]
    S = np.diag([z[0], z[1], z[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        S = S @ np.array([[-1, 0, 0, n[0] - 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    return S


def main():
    S, ses = sys.argv[1], sys.argv[2]
    num = S.split('-')[1]; fs = f'{FSD}/subj{num}'
    wd = f'{OUT}/{S}/{ses}'; os.makedirs(wd, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, ses, f'{time.time() - T0:6.0f}s', *x, flush=True)
    runs = sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/func/{S}_ses-{ses}_task-ring_run-*_bold.nii.gz'))
    evs = [r.replace('_bold.nii.gz', '_events.tsv') for r in runs]
    tr = json.load(open(runs[0].replace('.nii.gz', '.json')))['RepetitionTime']
    # ---- anatomy in FSL-readable form (FreeSurfer conformed space, scanner affine of orig.mgz)
    orig = nib.load(f'{fs}/mri/orig.mgz')
    t1 = f'{wd}/T1_brain.nii.gz'; wm = f'{wd}/wm.nii.gz'
    if not os.path.exists(wm):
        nib.save(nib.Nifti1Image(np.asarray(nib.load(f'{fs}/mri/brain.mgz').dataobj, np.float32), orig.affine), t1)
        aseg = np.asarray(nib.load(f'{fs}/mri/aseg.mgz').dataobj)
        nib.save(nib.Nifti1Image(np.isin(aseg, [2, 41]).astype(np.uint8), orig.affine), wm)
    done_glm = all(os.path.exists(f'{wd}/runbeta_{c}.nii.gz') for c in CONDS)
    # ---- motion correction (all runs to the mean of run 1 motion-corrected to itself)
    if not os.path.exists(f'{wd}/ref.nii.gz'):
        sh(f'mcflirt -in {runs[0]} -out {wd}/r1_self -meanvol -spline_final && fslmaths {wd}/r1_self -Tmean {wd}/ref && rm -f {wd}/r1_self.nii.gz')

    def mc(i):
        o = f'{wd}/mc_run-{i + 1:02d}'
        if not os.path.exists(o + '_mean.nii.gz'):
            sh(f'mcflirt -in {runs[i]} -reffile {wd}/ref -out {o} -spline_final -plots && fslmaths {o} -Tmean {o}_mean')
        return o
    if done_glm:
        mcs = [f'{wd}/mc_run-{i + 1:02d}' for i in range(len(runs))]
    else:
        with ThreadPoolExecutor(min(len(runs), 10)) as ex:
            mcs = list(ex.map(mc, range(len(runs))))
        sh(f'fslmerge -t {wd}/means ' + ' '.join(o + '_mean' for o in mcs) + f' && fslmaths {wd}/means -Tmean {wd}/session_mean && rm {wd}/means.nii.gz')
        say('motion correction done,', len(runs), 'runs')
    # ---- registration: FLIRT starts -> BBR on canonically reoriented copies (the EPI and FreeSurfer's conformed space are stored
    # with different axis orders, which FLIRT's search around identity cannot bridge); the lowest BBR cost is kept and converted
    # through world coordinates to the EPI voxel -> anatomical scanner RAS matrix
    mean = f'{wd}/session_mean.nii.gz'
    can = {}
    for nm, src in (('epi', mean), ('t1', t1), ('wm', wm)):
        img = nib.as_closest_canonical(nib.load(src)); can[nm] = f'{wd}/can_{nm}.nii.gz'; nib.save(img, can[nm])
    sch = '$FSLDIR/etc/flirtsch/measurecost1.sch'
    starts = {'search45': '-searchrx -45 45 -searchry -45 45 -searchrz -45 45',
              'search90': '-searchrx -90 90 -searchry -90 90 -searchrz -90 90 -coarsesearch 30 -finesearch 10',
              'nosearch': '-nosearch'}
    res = {}
    for name, opt in starts.items():
        try:
            sh(f"flirt -in {can['epi']} -ref {can['t1']} -dof 6 -cost corratio {opt} -omat {wd}/init_{name}.mat")
            sh(f"flirt -in {can['epi']} -ref {can['t1']} -dof 6 -cost bbr -wmseg {can['wm']} -init {wd}/init_{name}.mat -omat {wd}/bbr_{name}.mat "
               f"-schedule $FSLDIR/etc/flirtsch/bbr.sch")
            c = sh(f"flirt -in {can['epi']} -ref {can['t1']} -init {wd}/bbr_{name}.mat -cost bbr -wmseg {can['wm']} -schedule {sch} -omat /dev/null")
            res[name] = float(c.split()[0])
        except Exception as e:
            res[name] = float('inf'); say('start', name, 'failed', str(e)[-300:])
    best = min(res, key=res.get)
    M = np.loadtxt(f'{wd}/bbr_{best}.mat')
    ce, ct = nib.load(can['epi']), nib.load(can['t1']); epi = nib.load(mean)
    Tw = ct.affine @ np.linalg.inv(fsl_scaled(ct)) @ M @ fsl_scaled(ce) @ np.linalg.inv(ce.affine)   # EPI world -> anatomical world
    A_anat = Tw @ epi.affine                                                                        # EPI voxel -> anatomical scanner RAS
    np.savetxt(f'{wd}/epi2anat_vox2ras.txt', A_anat)
    for nm in can.values():
        os.remove(nm)
    say('registration: BBR costs', {k: round(v, 3) for k, v in res.items()}, 'best', best)
    # ---- mask and GLM per run
    if done_glm:
        qc = json.load(open(f'{wd}/{S}_{ses}_prep_qc.json')); qc.update(bbr_costs=res, bbr_best=best)
        json.dump(qc, open(f'{wd}/{S}_{ses}_prep_qc.json', 'w'), indent=1); say('registration redone; GLM kept'); return
    mv = np.asarray(epi.dataobj, np.float32)
    rmin = np.min([np.asarray(nib.load(o + '_mean.nii.gz').dataobj, np.float32) for o in mcs], axis=0)
    mask = (mv > 0.15 * np.percentile(mv[mv > 0], 99)) & (rmin >= 0.5 * mv)
    # sequential on purpose: concurrent lstsq calls from threads corrupted whole runs (thread-unsafe BLAS in the container)
    B = [glm_run(mcs[i] + '.nii.gz', mcs[i] + '.par', read_events(evs[i]), tr, mask) for i in range(len(runs))]
    for k, c in enumerate(CONDS):
        vol = np.zeros(mv.shape + (len(runs),), np.float32)
        for r in range(len(runs)):
            vol[..., r][mask] = B[r][k]
        nib.save(nib.Nifti1Image(vol, epi.affine, epi.header), f'{wd}/runbeta_{c}.nii.gz')
    nib.save(nib.Nifti1Image(np.stack([np.asarray(nib.load(o + '_mean.nii.gz').dataobj, np.float32) for o in mcs], -1), epi.affine, epi.header),
             f'{wd}/runmean.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), epi.affine), f'{wd}/mask.nii.gz')
    mot = np.concatenate([np.loadtxt(o + '.par') for o in mcs])
    qc = dict(subject=S, session=ses, n_runs=len(runs), tr=tr, bbr_costs=res, bbr_best=best, mask_voxels=int(mask.sum()),
              max_abs_translation_mm=float(np.abs(mot[:, 3:]).max()), max_abs_rotation_deg=float(np.degrees(np.abs(mot[:, :3]).max())),
              t_s=round(time.time() - T0))
    json.dump(qc, open(f'{wd}/{S}_{ses}_prep_qc.json', 'w'), indent=1)
    for o in mcs:
        os.remove(o + '.nii.gz')
    say('done', qc)


if __name__ == '__main__':
    main()
