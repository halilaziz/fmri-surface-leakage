"""Week 5, StudyForrest: per-run GLM betas for chosen sessions (fmriprep container: FSL + python), for within-session splits.
Week 4 kept only session / odd / even means. This re-runs motion correction exactly as week 4 did (every run to the session's
saved reference, spline) and the same GLM per run, and saves 4-D volumes (runs last) next to the week-4 outputs:
  runbetaL.nii.gz, runbetaR.nii.gz  percent signal change per run (inside the week-4 mask)
  runmean.nii.gz                    temporal mean of each motion-corrected run (for converting betas to raw units)
Check: the mean of the odd / even runs must reproduce week 4's betaL_odd_prep / betaL_even_prep.
Usage: python sf_runbetas.py SUBJECT [--sessions 7T-r14,7T-r20] [--jobs 10]
"""
import os, sys, json, glob, time, argparse, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_misregistration_studyforrest'))
from sf_prep import sh, glm_run, read_events, SF, OUT as PREP, ROOT

TMP = f'{ROOT}/work/tmp_runbetas'


def session_runs(S, key):
    field, res = key.split('-')
    if field == '7T':
        runs = sorted(glob.glob(f'{SF}/multires7t/{S}/ses-{res}/func/{S}_ses-{res}_task-orientation_rec-dico_run-*_bold.nii.gz'))
        pairs = [(r, r.replace('_rec-dico', '').replace('_bold.nii.gz', '_events.tsv')) for r in runs]
    else:
        runs = sorted(glob.glob(f'{SF}/multires3t/{S}/func/{S}_task-orientation_acq-{res}_run-*_bold.nii.gz'))
        pairs = [(r, r.replace('_bold.nii.gz', '_events.tsv')) for r in runs]
    return [(r, e) for r, e in pairs if os.path.exists(e)]          # as in sf_prep (sub-06 has a 0.8 mm run without events)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--sessions', default='7T-r14,7T-r20')
    ap.add_argument('--jobs', type=int, default=10)
    a = ap.parse_args(); S = a.subject
    qc_path = f'{PREP}/{S}/{S}.sf_runbetas_qc.json'
    qc = json.load(open(qc_path)) if os.path.exists(qc_path) else {}
    for key in a.sessions.split(','):
        wd = f'{PREP}/{S}/{key}'
        if os.path.exists(f'{wd}/runbetaR.nii.gz'):
            print(S, key, 'already done', flush=True); continue
        t0 = time.time()
        pairs = session_runs(S, key)
        mimg = nib.load(f'{wd}/session_mean.nii.gz')
        mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
        td = f'{TMP}/{S}/{key}'; os.makedirs(td, exist_ok=True)

        def one(i):
            r, e = pairs[i]
            o = f'{td}/mc_run-{i + 1:02d}'
            if not os.path.exists(o + '.nii.gz'):
                sh(f'mcflirt -in {r} -reffile {wd}/ref -out {o} -spline_final -plots')
            g = glm_run(o + '.nii.gz', o + '.par', read_events(e), mask)
            mean = np.asarray(nib.load(o + '.nii.gz').dataobj, np.float32).mean(3)
            par_same = bool(np.allclose(np.loadtxt(o + '.par'), np.loadtxt(f'{wd}/mc_run-{i + 1:02d}.par'), atol=1e-4))
            os.remove(o + '.nii.gz')
            return g, mean, par_same

        with ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(one, range(len(pairs))))
        out = {}
        for side in 'LR':
            vol = np.zeros(mimg.shape[:3] + (len(res),), np.float32)
            for k, (g, _, _) in enumerate(res):
                vol[..., k][mask] = g[side]
            out[side] = vol
            nib.save(nib.Nifti1Image(vol, mimg.affine, mimg.header), f'{wd}/runbeta{side}.nii.gz')
        nib.save(nib.Nifti1Image(np.stack([m for _, m, _ in res], -1), mimg.affine, mimg.header), f'{wd}/runmean.nii.gz')
        # reproducibility against week 4 (pre-fixmask odd / even means, inside the current mask)
        chk = {}
        for side in 'LR':
            for half, sl in (('odd', slice(0, None, 2)), ('even', slice(1, None, 2))):
                ref = np.asarray(nib.load(f'{wd}/beta{side}_{half}_prep.nii.gz').dataobj, np.float64)[mask]
                new = out[side][..., sl].mean(-1)[mask].astype(np.float64)
                chk[f'{side}_{half}_max_abs_diff'] = float(np.abs(new - ref).max())
                chk[f'{side}_{half}_rel_rms_diff'] = float(np.sqrt(np.mean((new - ref) ** 2) / max(np.mean(ref ** 2), 1e-30)))
        qc[key] = dict(n_runs=len(res), motion_params_identical=all(p for _, _, p in res), t_s=round(time.time() - t0), **chk)
        json.dump(qc, open(qc_path, 'w'), indent=1)
        shutil.rmtree(td, ignore_errors=True)
        print(S, key, qc[key], flush=True)


if __name__ == '__main__':
    main()
