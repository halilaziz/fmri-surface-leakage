"""Week 7, ds002702 stage 2: field-map distortion correction of one prepared session (kay_prep.py output), for the real
cross-session test (source and target sessions have different distortions; within-session emulation shares one).
  1. The session's GRE field map acquired closest in time to run 1 (the motion-correction reference): magnitude 1 brain
     (bet2, one erosion), fsl_prepare_fieldmap SIEMENS -> rad/s, resampled into the EPI grid through scanner coordinates.
  2. Unwarping along the phase-encoding axis (fugue; dwell = effective echo spacing; 2 mm field-map smoothing). The sign of the
     unwarp direction is chosen by the boundary-based registration cost of the unwarped session mean (lower is better).
  3. The per-run betas are unwarped in raw units together with the run means (the linear GLM commutes with a fixed spatial
     resampling), converted back to percent signal change; mask and registration (FLIRT + BBR as kay_prep.py) redone.
Output: results/kay/prep_dc/SUBJECT/SESSION with the same file names as kay_prep.py.
Usage (fmriprep container): python3 kay_undistort.py SUBJECT SESSION
"""
import os, sys, json, glob, time, shutil
import numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kay_prep import G, DATA, OUT as PREP, CONDS, sh, fsl_scaled

OUT = f'{G}/results/kay/prep_dc'


def tsec(s):
    h, m, x = s.split(':')
    return 3600 * int(h) + 60 * int(m) + float(x)


def register(wd, mean, t1, wm, tag):
    """FLIRT (3 starts) + BBR on canonically reoriented copies, as kay_prep.py; returns (EPI voxel -> anatomical RAS, costs)."""
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
    S, ses = sys.argv[1], sys.argv[2]
    src = f'{PREP}/{S}/{ses}'; wd = f'{OUT}/{S}/{ses}'; os.makedirs(wd, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, ses, f'{time.time() - T0:6.0f}s', *x, flush=True)
    bj = json.load(open(sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/func/{S}_ses-{ses}_task-ring_run-*_bold.json'))[0]))
    esp = float(bj['EffectiveEchoSpacing']); pe = bj['PhaseEncodingDirection']
    ax = {'i': 'x', 'j': 'y', 'k': 'z'}[pe[0]]
    fm = sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/fmap/{S}_ses-{ses}_run-*_phasediff.json'))
    k = int(np.argmin([abs(tsec(json.load(open(p))['AcquisitionTime']) - tsec(bj['AcquisitionTime'])) for p in fm]))
    pd = fm[k].replace('.json', '.nii.gz'); m1 = pd.replace('phasediff', 'magnitude1')
    te1 = json.load(open(m1.replace('.nii.gz', '.json')))['EchoTime']; te2 = json.load(open(fm[k]))['EchoTime']
    dte_ms = 1000 * (te2 - te1)
    # ---- field map (rad/s) in the EPI grid
    sh(f'bet2 {m1} {wd}/mag_brain -f 0.5 && fslmaths {wd}/mag_brain -ero {wd}/mag_brain_ero')      # bet -R needs dc (absent)
    sh(f'fsl_prepare_fieldmap SIEMENS {pd} {wd}/mag_brain_ero {wd}/fmap_rads {dte_ms:.4f}')
    fimg = nib.load(f'{wd}/fmap_rads.nii.gz'); F = np.asarray(fimg.dataobj, np.float64)
    mimg = nib.load(f'{src}/session_mean.nii.gz'); shp = mimg.shape[:3]
    ijk = np.stack(np.meshgrid(*[np.arange(d) for d in shp], indexing='ij'), -1).reshape(-1, 3).astype(np.float64)
    q = (np.linalg.inv(fimg.affine) @ mimg.affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3]
    Fe = map_coordinates(F, q, order=1, mode='nearest').reshape(shp).astype(np.float32)
    nib.save(nib.Nifti1Image(Fe, mimg.affine, mimg.header), f'{wd}/fmap_epi.nii.gz')
    npe = shp['xyz'.index(ax)]
    shift_vox = Fe / (2 * np.pi) * esp * npe
    say('field map', os.path.basename(pd), f'dTE {dte_ms:.2f} ms, esp {1000 * esp:.3f} ms, PE {pe} ({npe} lines)')
    # ---- unwarp direction by registration cost
    t1, wm = f'{src}/T1_brain.nii.gz', f'{src}/wm.nii.gz'
    reg = {}
    for d in (ax, ax + '-'):
        o = f"{wd}/mean_u{'m' if d.endswith('-') else 'p'}"
        sh(f'fugue -i {src}/session_mean --loadfmap={wd}/fmap_epi --dwell={esp} --unwarpdir={d} --smooth3=2 -u {o}')
        reg[d] = register(wd, o + '.nii.gz', t1, wm, 'm' if d.endswith('-') else 'p')
        say('unwarp', d, 'BBR costs', {k_: round(v, 3) for k_, v in reg[d][1].items()})
    d = min(reg, key=lambda x: min(reg[x][1].values()))
    A_anat, costs, best = reg[d]
    tag = 'm' if d.endswith('-') else 'p'
    shutil.move(f'{wd}/mean_u{tag}.nii.gz', f'{wd}/session_mean.nii.gz')
    # ---- betas: unwarp raw-unit betas and run means with the chosen direction, back to percent signal change
    fu = lambda i, o: sh(f'fugue -i {i} --loadfmap={wd}/fmap_epi --dwell={esp} --unwarpdir={d} --smooth3=2 -u {o}')
    fu(f'{src}/runmean', f'{wd}/runmean')
    rm = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    rm0 = np.asarray(nib.load(f'{src}/runmean.nii.gz').dataobj, np.float32)
    mv = np.asarray(nib.load(f'{wd}/session_mean.nii.gz').dataobj, np.float32)
    mask = (mv > 0.15 * np.percentile(mv[mv > 0], 99)) & (rm.min(-1) >= 0.5 * mv)
    for c in CONDS:
        b = nib.load(f'{src}/runbeta_{c}.nii.gz'); B = np.asarray(b.dataobj, np.float32)
        nib.save(nib.Nifti1Image(B * rm0 / 100.0, b.affine, b.header), f'{wd}/raw_{c}.nii.gz')
        fu(f'{wd}/raw_{c}', f'{wd}/rawu_{c}')
        Ru = np.asarray(nib.load(f'{wd}/rawu_{c}.nii.gz').dataobj, np.float32)
        P = np.where(mask[..., None] & (rm > 0), 100.0 * Ru / np.where(rm > 0, rm, 1.0), 0).astype(np.float32)
        nib.save(nib.Nifti1Image(P, b.affine, b.header), f'{wd}/runbeta_{c}.nii.gz')
        os.remove(f'{wd}/raw_{c}.nii.gz'); os.remove(f'{wd}/rawu_{c}.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), mimg.affine), f'{wd}/mask.nii.gz')
    np.savetxt(f'{wd}/epi2anat_vox2ras.txt', A_anat)
    qc = json.load(open(f'{src}/{S}_{ses}_prep_qc.json'))
    qc.update(stage='field-map corrected', fieldmap=os.path.basename(pd), dte_ms=dte_ms, unwarpdir=d,
              bbr_costs=costs, bbr_best=best, bbr_costs_other_direction=reg[[x for x in reg if x != d][0]][1],
              bbr_costs_uncorrected=qc.get('bbr_costs'), mask_voxels=int(mask.sum()),
              shift_vox_abs_p50_p95=[float(np.percentile(np.abs(shift_vox[mask]), p)) for p in (50, 95)],
              t_s=round(time.time() - T0))
    json.dump(qc, open(f'{wd}/{S}_{ses}_prep_qc.json', 'w'), indent=1)
    for p in glob.glob(f'{wd}/mean_u*.nii.gz'):
        os.remove(p)
    say('done', {k_: qc[k_] for k_ in ('unwarpdir', 'bbr_costs', 'bbr_costs_uncorrected', 'shift_vox_abs_p50_p95')})


if __name__ == '__main__':
    main()
