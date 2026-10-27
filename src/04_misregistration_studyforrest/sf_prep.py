"""Week 4, StudyForrest multi-resolution data: preprocessing of one subject (run inside the fmriprep container: FSL, FreeSurfer,
python with nibabel/numpy/scipy).

Per session (7T 0.8 / 1.4 / 2.0 / 3.0 mm; 3T 1.4 / 2.0 / 3.0 mm where available), on the NATIVE acquisition grid (no resampling
beyond motion correction):
  1. motion correction: mcflirt of every run to a session reference (the temporal mean of motion-corrected run 1), spline;
  2. registration of the session mean to the subject's FreeSurfer anatomy: FLIRT (6 dof, correlation ratio, rotation search) for
     initialisation, then bbregister (BBR, 6 dof). 0.8 mm slab: initialised from the same session's large-coverage 0.8 mm run.
     3T: initialised through the same session's T1w;
  3. GLM per run: left- and right-hemifield stimulation (canonical double-gamma HRF + temporal derivative, 3 s trials), Legendre
     drifts (order 3) and 6 motion parameters; betas in percent signal change; session mean, variance of the mean over runs and
     odd/even halves;
  4. QC: bbregister cost, motion summary.
Usage: python sf_prep.py SUBJECT (e.g. sub-06) [--sessions 7T-r14,7T-r20,...]
"""
import os, sys, json, glob, time, shutil, argparse, subprocess
import numpy as np
import nibabel as nib
from scipy.stats import gamma

ROOT = os.environ.get('GEO_ROOT', '/path/to/geo_root')
SF = f'{ROOT}/data/studyforrest'
OUT = f'{ROOT}/results/week4/sf/prep'
os.environ['SUBJECTS_DIR'] = f'{SF}/freesurfer'
TR = 2.0


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}\nSTDOUT: {r.stdout[-2500:]}\nSTDERR: {r.stderr[-1500:]}')
    return r.stdout


def hrf(tr, n=32):
    """SPM canonical double-gamma HRF (peak 6 s, undershoot 16 s, ratio 1/6) and its temporal derivative, sampled at tr/16."""
    dt = tr / 16; t = np.arange(0, n, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return t, h / h.sum(), dt


def design(events, nvol, tr):
    """Columns: left-hemifield stimulus, right-hemifield stimulus, their temporal derivatives (fine-grid convolution)."""
    t, h, dt = hrf(tr)
    tt = np.arange(0, nvol * tr, dt)
    cols = []
    for side in ('lh_orientation', 'rh_orientation'):
        box = np.zeros(len(tt))
        for on, du, lab in zip(events['onset'], events['duration'], events[side]):
            if lab != 'none':
                box[(tt >= on) & (tt < on + du)] = 1.0
        x = np.convolve(box, h)[:len(tt)]
        dx = np.gradient(x)
        idx = (np.arange(nvol) * tr / dt + 0.5 * tr / dt).astype(int)   # mid-TR samples (ascending slices span the TR)
        cols += [x[idx], dx[idx]]
    return np.stack(cols, 1)


def read_events(p):
    lines = open(p).read().strip().split('\n'); hdr = lines[0].split('\t')
    rows = [l.split('\t') for l in lines[1:]]
    ev = {k: [r[i] for r in rows] for i, k in enumerate(hdr)}
    ev['onset'] = np.array(ev['onset'], float); ev['duration'] = np.array(ev['duration'], float)
    return ev


def legendre(nvol, order=3):
    x = np.linspace(-1, 1, nvol)
    return np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(order + 1)], 1)


def glm_run(bold, par, events, mask):
    img = nib.load(bold); Y = np.asarray(img.dataobj, np.float32)[mask]          # voxels x time
    nvol = Y.shape[1]
    X = np.concatenate([design(events, nvol, TR), legendre(nvol), np.loadtxt(par)[:nvol]], 1)
    B, *_ = np.linalg.lstsq(X, Y.T.astype(np.float64), rcond=None)
    mean = np.maximum(Y.mean(1), 1e-6)
    res = Y.T - X @ B
    dof = nvol - X.shape[1]
    s2 = (res ** 2).sum(0) / dof
    XtXi = np.linalg.inv(X.T @ X)
    pct = lambda b: 100 * b / mean
    return dict(L=pct(B[0]), R=pct(B[2]), varL=(100 / mean) ** 2 * s2 * XtXi[0, 0], varR=(100 / mean) ** 2 * s2 * XtXi[2, 2])


def mcflirt_session(runs, wd):
    """Reference = temporal mean of run 1 motion-corrected to its own middle volume; all runs to that reference."""
    os.makedirs(wd, exist_ok=True)
    sh(f'mcflirt -in {runs[0]} -out {wd}/r1_self -meanvol -spline_final')
    sh(f'fslmaths {wd}/r1_self -Tmean {wd}/ref')
    out = []
    for i, r in enumerate(runs):
        o = f'{wd}/mc_run-{i + 1:02d}'
        if not os.path.exists(o + '.nii.gz'):
            sh(f'mcflirt -in {r} -reffile {wd}/ref -out {o} -spline_final -plots')
        out.append(o)
    means = []
    for o in out:
        sh(f'fslmaths {o} -Tmean {o}_mean'); means.append(f'{o}_mean')
    sh(f'fslmerge -t {wd}/means ' + ' '.join(means) + f' && fslmaths {wd}/means -Tmean {wd}/session_mean && rm {wd}/means.nii.gz')
    return out


def register(S, mov, wd, t1brain, init_lta=None):
    """mov (session mean, NIfTI) -> FreeSurfer anatomy (bbregister, 6 dof). Without init_lta, FLIRT (correlation ratio,
    +-45 deg search) initialises. Returns (lta path, bbregister min cost)."""
    if init_lta is None:
        sh(f'flirt -in {mov} -ref {t1brain} -dof 6 -cost corratio -searchrx -45 45 -searchry -45 45 -searchrz -45 45 '
           f'-omat {wd}/flirt_init.mat')
        sh(f'lta_convert --infsl {wd}/flirt_init.mat --src {mov} --trg {t1brain} --outlta {wd}/init.lta --subject {S}')
        init_lta = f'{wd}/init.lta'
    # --init-reg resets the subject from the file, so --s comes after it (and the LTAs carry the subject name)
    sh(f'bbregister --mov {mov} --init-reg {init_lta} --s {S} --bold --6 --reg {wd}/bbr.lta')
    cost = float(open(glob.glob(f'{wd}/bbr*.mincost')[0]).read().split()[0])   # bbregister writes bbr.dat.mincost
    return f'{wd}/bbr.lta', cost


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--sessions', default='')
    a = ap.parse_args(); S = a.subject
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    T0 = time.time(); qc_path = f'{od}/{S}.sf_prep_qc.json'
    qc = json.load(open(qc_path)) if os.path.exists(qc_path) else {'subject': S, 'sessions': {}}     # keep finished sessions' entries
    qc.setdefault('sessions', {})
    t1brain = f'{od}/T1_brain.nii.gz'
    if not os.path.exists(t1brain):
        sh(f'mri_convert {SF}/freesurfer/{S}/mri/brain.mgz {t1brain}')
    orig = f'{SF}/freesurfer/{S}/mri/orig.mgz'
    sessions = []
    for res in ('r08', 'r14', 'r20', 'r30'):
        runs = sorted(glob.glob(f'{SF}/multires7t/{S}/ses-{res}/func/{S}_ses-{res}_task-orientation_rec-dico_run-*_bold.nii.gz'))
        pairs = [(r, r.replace('_rec-dico', '').replace('_bold.nii.gz', '_events.tsv')) for r in runs]
        pairs = [(r, e) for r, e in pairs if os.path.exists(e)]          # e.g. sub-06 has an 11th 0.8 mm run without events
        if pairs:
            sessions.append(('7T', res, [r for r, _ in pairs], [e for _, e in pairs]))
    for res in ('r14', 'r20', 'r30'):
        runs = sorted(glob.glob(f'{SF}/multires3t/{S}/func/{S}_task-orientation_acq-{res}_run-*_bold.nii.gz'))
        if runs:
            sessions.append(('3T', res, runs, [r.replace('_bold.nii.gz', '_events.tsv') for r in runs]))
    if a.sessions:
        keep = set(a.sessions.split(','))
        sessions = [s for s in sessions if f'{s[0]}-{s[1]}' in keep]
    print(S, 'sessions:', [f'{f}-{r} ({len(x)} runs)' for f, r, x, _ in sessions], flush=True)
    for field, res, runs, evs in sessions:
        key = f'{field}-{res}'; wd = f'{od}/{key}'; os.makedirs(wd, exist_ok=True)
        if os.path.exists(f'{wd}/betaR_even.nii.gz'):
            if 'bbr_cost' not in qc['sessions'].get(key, {}):
                qc['sessions'][key] = rebuild_qc(wd)
            print(S, key, 'already done', qc['sessions'][key].get('bbr_cost'), flush=True); continue
        try:
            prep_session(S, field, res, runs, evs, wd, t1brain, orig, qc)
        except Exception as e:                      # one failed session must not stop the others
            qc['sessions'][key] = {'error': str(e)[-1500:]}
            print(S, key, 'FAILED', str(e)[-600:], flush=True)
    qc['t_total_s'] = round(time.time() - T0)
    json.dump(qc, open(qc_path, 'w'), indent=1)


def rebuild_qc(wd):
    """QC entry of a finished session from its files: first-pass BBR cost (bbregister's bbr.dat.mincost) and motion parameters."""
    q = {'rebuilt_from_files': True}
    for k, f in (('bbr_cost', f'{wd}/bbr.dat.mincost'), ('coverage_bbr_cost', f'{wd}/coverage/bbr.dat.mincost')):
        if os.path.exists(f):
            q[k] = float(open(f).read().split()[0])
    pars = sorted(glob.glob(f'{wd}/mc_run-*.par'))
    if pars:
        mot = np.concatenate([np.loadtxt(p) for p in pars])
        q.update(n_runs=len(pars), max_abs_translation_mm=float(np.abs(mot[:, 3:]).max()),
                 max_abs_rotation_deg=float(np.degrees(np.abs(mot[:, :3]).max())))
    return q


def prep_session(S, field, res, runs, evs, wd, t1brain, orig, qc):
        key = f'{field}-{res}'
        t0 = time.time()
        mc = mcflirt_session(runs, wd)
        mean = f'{wd}/session_mean.nii.gz'
        if field == '7T' and res == 'r08':
            cov = glob.glob(f'{SF}/multires7t/{S}/ses-r08/func/{S}_ses-r08_task-coverage_rec-dico_bold.nii.gz')[0]
            cw = f'{wd}/coverage'; os.makedirs(cw, exist_ok=True)
            sh(f'mcflirt -in {cov} -out {cw}/cov_mc -meanvol -spline_final && fslmaths {cw}/cov_mc -Tmean {cw}/cov_mean')
            cov_lta, cov_cost = register(S, f'{cw}/cov_mean.nii.gz', cw, t1brain)
            # slab and coverage run share the session's scanner space: the coverage transform (RAS to RAS) with the slab as source
            sh(f'lta_convert --inlta {cov_lta} --src {mean} --trg {orig} --outlta {wd}/init.lta --subject {S}')
            lta, cost = register(S, mean, wd, t1brain, init_lta=f'{wd}/init.lta')
            qc['sessions'][key] = {'coverage_bbr_cost': cov_cost}
        elif field == '3T' and glob.glob(f'{SF}/multires3t/{S}/anat/{S}_acq-{res}_T1w.nii.gz'):
            # EPI and the in-session T1w share scanner space: T1w -> FreeSurfer brain (FLIRT, MI), EPI -> T1w (no search), composed
            t1s = glob.glob(f'{SF}/multires3t/{S}/anat/{S}_acq-{res}_T1w.nii.gz')[0]
            sh(f'flirt -in {t1s} -ref {t1brain} -dof 6 -cost mutualinfo -searchrx -30 30 -searchry -30 30 -searchrz -30 30 -omat {wd}/t1s2fs.mat')
            sh(f'flirt -in {mean} -ref {t1s} -dof 6 -cost corratio -nosearch -omat {wd}/epi2t1s.mat')
            sh(f'convert_xfm -omat {wd}/flirt_init.mat -concat {wd}/t1s2fs.mat {wd}/epi2t1s.mat')
            sh(f'lta_convert --infsl {wd}/flirt_init.mat --src {mean} --trg {t1brain} --outlta {wd}/init.lta --subject {S}')
            lta, cost = register(S, mean, wd, t1brain, init_lta=f'{wd}/init.lta')
            qc['sessions'][key] = {}
        else:
            lta, cost = register(S, mean, wd, t1brain)
            qc['sessions'][key] = {}
        # brain mask on the session grid (mean above 15% of its robust maximum)
        mimg = nib.load(mean); mv = np.asarray(mimg.dataobj, np.float32)
        mask = mv > 0.15 * np.percentile(mv[mv > 0], 99)
        # slab-edge voxels zero-filled in some run after motion correction make the percent-change normalisation explode
        rmin = np.min([np.asarray(nib.load(o + '_mean.nii.gz').dataobj, np.float32) for o in mc], axis=0)
        mask &= rmin >= 0.5 * mv
        per = [glm_run(o + '.nii.gz', o + '.par', read_events(e), mask) for o, e in zip(mc, evs)]
        outv = {}
        for side in 'LR':
            Bm = np.stack([p[side] for p in per]); nr = len(per)
            outv[f'beta{side}'] = Bm.mean(0)
            outv[f'var{side}'] = Bm.var(0, ddof=1) / nr                       # variance of the mean from run-to-run spread
            outv[f'beta{side}_odd'] = Bm[0::2].mean(0); outv[f'beta{side}_even'] = Bm[1::2].mean(0)
        for k, v in outv.items():
            vol = np.zeros(mv.shape, np.float32); vol[mask] = v
            nib.save(nib.Nifti1Image(vol, mimg.affine, mimg.header), f'{wd}/{k}.nii.gz')
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), mimg.affine), f'{wd}/mask.nii.gz')
        mot = np.concatenate([np.loadtxt(o + '.par') for o in mc])
        qc['sessions'][key].update(dict(bbr_cost=cost, n_runs=len(mc), max_abs_translation_mm=float(np.abs(mot[:, 3:]).max()),
                                        max_abs_rotation_deg=float(np.degrees(np.abs(mot[:, :3]).max())),
                                        t_s=round(time.time() - t0)))
        for o in mc:                    # keep parameters and means; drop the motion-corrected series (large, not needed later)
            os.remove(o + '.nii.gz')
        for f in glob.glob(f'{wd}/r1_self*.nii.gz') + glob.glob(f'{wd}/coverage/cov_mc.nii.gz'):
            os.remove(f)
        print(S, key, 'done', qc['sessions'][key], flush=True)


if __name__ == '__main__':
    main()
