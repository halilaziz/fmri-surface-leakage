"""Week 17, NYU Retinotopy (OpenNeuro ds003787): preprocessing of one subject on its native 2.0 mm grid (fMRIPrep container:
FSL + python), as PLAN_NYU.md:
  1. topup on the AP/PA spin-echo pair (phase-encoding vectors and readout times from the sidecars) -> field;
  2. motion correction of every run to the run-1 single-band reference, itself registered rigidly (6 dof, normalised mutual
     information) to the P->A spin-echo image and resampled once into its grid; mcflirt with spline output;
  3. distortion correction of every run with applytopup (the P->A line, Jacobian modulation, spline);
  4. registration of the corrected session mean to the FreeSurfer anatomy of the dataset: FLIRT from three starts, then
     boundary-based registration with the FreeSurfer white matter, lowest cost kept (as ds002702, kay_prep.py);
  5. brain mask (as ds002702); runs whose maximum absolute translation exceeds 4 mm are excluded (plan).
Output: GEO_CONST/results/nyu/prep/SUBJECT: dc_run-XX.nii.gz (corrected series, float32), session_mean, mask, epi2anat_vox2ras.txt,
motion parameters, and SUBJECT_prep_qc.json. Only files inside GEO_CONST are written.
Usage (fmriprep container): python3 nyu_prep.py SUBJECT [--threads N]
"""
import os, sys, json, glob, time, shutil, subprocess, argparse
import numpy as np, nibabel as nib
from concurrent.futures import ThreadPoolExecutor

G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
DATA = f'{G}/data/ds003787'
FSD = f'{DATA}/derivatives/freesurfer'
OUT = f'{G}/results/nyu/prep'
SES = 'ses-nyu3t01'
MAX_TRANS_MM = 4.0


def sh(cmd, expect=None):
    """Run a shell command; raise on a non-zero exit, on 'Exception'/'Error' in its output (FSL tools sometimes exit 0 after
    failing to read an image), or when an expected output file is missing."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    bad = r.returncode != 0 or 'Image Exception' in r.stdout + r.stderr or (expect is not None and not os.path.exists(expect))
    if bad:
        raise RuntimeError(f'failed ({r.returncode}): {cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return r.stdout


def fsl_scaled(img):
    """FSL's 'scaled voxel' coordinates: voxel * pixdim, with x flipped when the affine has a positive determinant."""
    n = img.shape; z = img.header.get_zooms()[:3]
    S = np.diag([z[0], z[1], z[2], 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        S = S @ np.array([[-1, 0, 0, n[0] - 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    return S


def pe_line(js):
    """topup acquisition line from a BIDS sidecar: phase-encoding vector in voxel axes and the total readout time."""
    pe = js['PhaseEncodingDirection']; v = [0, 0, 0]; v['ijk'.index(pe[0])] = -1 if pe.endswith('-') else 1
    return v, float(js['TotalReadoutTime'])


def register(wd, mean, t1, wm, tag):
    """FLIRT (3 starts) + BBR on canonically reoriented copies (kay_undistort.register); returns (EPI voxel -> anatomical RAS, costs, best)."""
    can = {}
    for nm, src in (('epi', mean), ('t1', t1), ('wm', wm)):
        can[nm] = f'{wd}/can_{tag}_{nm}.nii.gz'; nib.save(nib.as_closest_canonical(nib.load(src)), can[nm])
    sch = '$FSLDIR/etc/flirtsch/measurecost1.sch'
    starts = {'search45': '-searchrx -45 45 -searchry -45 45 -searchrz -45 45',
              'search90': '-searchrx -90 90 -searchry -90 90 -searchrz -90 90 -coarsesearch 30 -finesearch 10', 'nosearch': '-nosearch'}
    res = {}
    for name, opt in starts.items():
        try:
            sh(f"flirt -in {can['epi']} -ref {can['t1']} -dof 6 -cost corratio {opt} -omat {wd}/init_{tag}_{name}.mat")
            sh(f"flirt -in {can['epi']} -ref {can['t1']} -dof 6 -cost bbr -wmseg {can['wm']} -init {wd}/init_{tag}_{name}.mat "
               f"-omat {wd}/bbr_{tag}_{name}.mat -schedule $FSLDIR/etc/flirtsch/bbr.sch")
            c = sh(f"flirt -in {can['epi']} -ref {can['t1']} -init {wd}/bbr_{tag}_{name}.mat -cost bbr -wmseg {can['wm']} -schedule {sch} -omat /dev/null")
            res[name] = float(c.split()[0])
        except Exception:
            res[name] = float('inf')
    best = min(res, key=res.get)
    M = np.loadtxt(f'{wd}/bbr_{tag}_{best}.mat')
    ce, ct, epi = nib.load(can['epi']), nib.load(can['t1']), nib.load(mean)
    A = ct.affine @ np.linalg.inv(fsl_scaled(ct)) @ M @ fsl_scaled(ce) @ np.linalg.inv(ce.affine) @ epi.affine
    for p in can.values():
        os.remove(p)
    return A, res, best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args(); S = a.subject
    src = f'{DATA}/{S}/{SES}'; wd = f'{OUT}/{S}'; os.makedirs(wd, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, f'{time.time() - T0:6.0f}s', *x, flush=True)
    runs = sorted(glob.glob(f'{src}/func/{S}_{SES}_task-prf*_run-*_bold.nii.gz'), key=lambda r: os.path.basename(r).split('_run-')[1])   # with or without acq-PA
    rid = [os.path.basename(r).split('_run-')[1][:2] for r in runs]
    sbref1 = runs[0].replace('_bold.nii.gz', '_sbref.nii.gz')
    if not os.path.exists(sbref1):                     # some runs come without a single-band reference: mean of run 1's first 10 volumes
        sbref1 = f'{wd}/run1_first10_mean.nii.gz'
        if not os.path.exists(sbref1):
            sh(f'fslroi {runs[0]} {wd}/run1_first10 0 10 && fslmaths {wd}/run1_first10 -Tmean {sbref1} && rm -f {wd}/run1_first10.nii.gz')
    bj = json.load(open(runs[0].replace('.nii.gz', '.json')))
    fm = {d: sorted(glob.glob(f'{src}/fmap/{S}_{SES}_acq-fMRI_dir-{d}*_epi.nii.gz')) for d in ('AP', 'PA')}                 # with or without run-01
    assert fm['AP'] and fm['PA'], 'no AP/PA spin-echo pair'
    fm = {d: fm[d][0] for d in fm}
    qc = dict(subject=S, runs=rid, bold_pe=bj['PhaseEncodingDirection'], bold_trt=bj['TotalReadoutTime'], fieldmaps={d: os.path.basename(p) for d, p in fm.items()})
    # ---- 1. topup on the spin-echo pair
    if not os.path.exists(f'{wd}/topup_fieldcoef.nii.gz'):
        lines, imgs = [], []
        for d in ('AP', 'PA'):
            img = nib.load(fm[d]); nv = img.shape[3] if img.ndim == 4 else 1
            v, trt = pe_line(json.load(open(fm[d].replace('.nii.gz', '.json'))))
            lines += [f'{v[0]} {v[1]} {v[2]} {trt:.6f}'] * nv; imgs.append(fm[d])
            qc[f'n_vol_{d}'] = nv
        open(f'{wd}/acqparams.txt', 'w').write('\n'.join(lines) + '\n')
        sh(f'fslmerge -t {wd}/se_merged ' + ' '.join(imgs))
        sh(f'topup --imain={wd}/se_merged --datain={wd}/acqparams.txt --config=b02b0.cnf --out={wd}/topup --iout={wd}/se_corr --fout={wd}/field_hz')
        say('topup done', qc.get('n_vol_AP'), '+', qc.get('n_vol_PA'), 'volumes')
    lines = open(f'{wd}/acqparams.txt').read().split('\n')
    bv, _ = pe_line(bj); bline = f'{bv[0]} {bv[1]} {bv[2]}'
    pa_index = 1 + next(i for i, L in enumerate(lines) if L.startswith(bline))        # first spin-echo volume with the BOLD's direction
    qc['applytopup_index'] = pa_index
    # ---- 2. reference: run-1 SBRef registered to that spin-echo volume, resampled once into its grid
    if not os.path.exists(f'{wd}/ref.nii.gz'):
        sh(f'fslroi {wd}/se_merged {wd}/se_ref {pa_index - 1} 1')
        sh(f'flirt -in {sbref1} -ref {wd}/se_ref -dof 6 -cost normmi -searchrx -10 10 -searchry -10 10 -searchrz -10 10 -omat {wd}/sbref2se.mat', expect=f'{wd}/sbref2se.mat')
        sh(f'flirt -in {sbref1} -ref {wd}/se_ref -applyxfm -init {wd}/sbref2se.mat -interp spline -out {wd}/ref', expect=f'{wd}/ref.nii.gz')
        qc['reference'] = os.path.basename(sbref1)
    # ---- 2-3. motion correction into the reference grid, then applytopup, per run
    def one(i):
        o = f'{wd}/mc_run-{rid[i]}'; dc = f'{wd}/dc_run-{rid[i]}.nii.gz'
        if not os.path.exists(dc):
            sh(f'mcflirt -in {runs[i]} -reffile {wd}/ref -out {o} -spline_final -plots')
            sh(f'applytopup --imain={o} --inindex={pa_index} --datain={wd}/acqparams.txt --topup={wd}/topup --method=jac --interp=spline --out={wd}/dc_tmp_{rid[i]}')
            img = nib.load(f'{wd}/dc_tmp_{rid[i]}.nii.gz')
            nib.save(nib.Nifti1Image(np.asarray(img.dataobj, np.float32), img.affine, img.header), dc)
            os.remove(f'{wd}/dc_tmp_{rid[i]}.nii.gz'); os.remove(o + '.nii.gz')
        if not os.path.exists(f'{wd}/dcmean_run-{rid[i]}.nii.gz'):
            sh(f'fslmaths {dc} -Tmean {wd}/dcmean_run-{rid[i]}')
        par = np.loadtxt(o + '.par')
        return float(np.abs(par[:, 3:]).max()), float(np.degrees(np.abs(par[:, :3]).max()))
    with ThreadPoolExecutor(min(len(runs), a.threads)) as ex:
        mot = list(ex.map(one, range(len(runs))))
    qc['max_abs_translation_mm'] = {r: m[0] for r, m in zip(rid, mot)}; qc['max_abs_rotation_deg'] = {r: m[1] for r, m in zip(rid, mot)}
    keep = [r for r, m in zip(rid, mot) if m[0] <= MAX_TRANS_MM]; qc['runs_kept'] = keep; qc['runs_excluded'] = [r for r in rid if r not in keep]
    say('motion + distortion correction done;', len(keep), 'of', len(rid), 'runs kept; max translation', round(max(m[0] for m in mot), 2), 'mm')
    sh(f'fslmerge -t {wd}/means ' + ' '.join(f'{wd}/dcmean_run-{r}' for r in keep) + f' && fslmaths {wd}/means -Tmean {wd}/session_mean && rm {wd}/means.nii.gz')
    # ---- 4. registration to the FreeSurfer anatomy (scanner RAS of orig.mgz)
    fs = f'{FSD}/{S}'
    orig = nib.load(f'{fs}/mri/orig.mgz'); t1 = f'{wd}/T1_brain.nii.gz'; wm = f'{wd}/wm.nii.gz'
    if not os.path.exists(wm):
        nib.save(nib.Nifti1Image(np.asarray(nib.load(f'{fs}/mri/brain.mgz').dataobj, np.float32), orig.affine), t1)
        aseg = np.asarray(nib.load(f'{fs}/mri/aseg.mgz').dataobj)
        nib.save(nib.Nifti1Image(np.isin(aseg, [2, 41]).astype(np.uint8), orig.affine), wm)
    A_anat, costs, best = register(wd, f'{wd}/session_mean.nii.gz', t1, wm, 'dc')
    np.savetxt(f'{wd}/epi2anat_vox2ras.txt', A_anat)
    qc.update(bbr_costs=costs, bbr_best=best, bbr_ok=bool(min(costs.values()) <= 0.8))
    say('registration: BBR costs', {k: round(v, 3) for k, v in costs.items()}, 'best', best)
    # ---- 5. mask (as ds002702)
    mimg = nib.load(f'{wd}/session_mean.nii.gz'); mv = np.asarray(mimg.dataobj, np.float32)
    rmin = np.min([np.asarray(nib.load(f'{wd}/dcmean_run-{r}.nii.gz').dataobj, np.float32) for r in keep], axis=0)
    mask = (mv > 0.15 * np.percentile(mv[mv > 0], 99)) & (rmin >= 0.5 * mv)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), mimg.affine), f'{wd}/mask.nii.gz')
    fld = np.asarray(nib.load(f'{wd}/field_hz.nii.gz').dataobj, np.float32)
    qc.update(mask_voxels=int(mask.sum()), field_hz_abs_p50_p95=[float(np.percentile(np.abs(fld), p)) for p in (50, 95)],
              included=bool(len(keep) >= 4 and qc['bbr_ok']), t_s=round(time.time() - T0))
    json.dump(qc, open(f'{wd}/{S}_prep_qc.json', 'w'), indent=1)
    for p in glob.glob(f'{wd}/dcmean_run-*.nii.gz') + [f'{wd}/se_merged.nii.gz']:
        if os.path.exists(p):
            os.remove(p)
    say('done', {k: qc[k] for k in ('runs_kept', 'bbr_best', 'included', 'field_hz_abs_p50_p95')})


if __name__ == '__main__':
    main()
