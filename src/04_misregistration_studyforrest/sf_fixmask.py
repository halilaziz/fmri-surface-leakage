"""Week 4, StudyForrest: drop slab-edge voxels whose mean in some run is low (zero-filled after motion correction), where the
percent-signal-change normalisation by the run mean explodes (betas up to 1e8 % were found). A voxel is kept only if its mean in
every run is at least half its session mean. Updates mask.nii.gz and zeroes betas / variances outside it (originals kept as *_prep).
Usage: python sf_fixmask.py SUBJECT
"""
import os, sys, glob, shutil
import numpy as np, nibabel as nib

PREP = os.environ.get('GEO_ROOT', '/path/to/geo_root') + '/results/week4/sf/prep'
S = sys.argv[1]
for wd in sorted(glob.glob(f'{PREP}/{S}/*T-r*')):
    if not os.path.exists(f'{wd}/betaR_even.nii.gz'):
        continue
    if not os.path.exists(f'{wd}/mask_prep.nii.gz'):
        shutil.copy(f'{wd}/mask.nii.gz', f'{wd}/mask_prep.nii.gz')
    mimg = nib.load(f'{wd}/session_mean.nii.gz'); smean = np.asarray(mimg.dataobj, float)
    runs = sorted(glob.glob(f'{wd}/mc_run-*_mean.nii.gz'))
    rmin = np.min([np.asarray(nib.load(r).dataobj, float) for r in runs], axis=0)
    mask0 = np.asarray(nib.load(f'{wd}/mask_prep.nii.gz').dataobj) > 0
    mask = mask0 & (rmin >= 0.5 * smean)
    for n in ('betaL', 'betaR', 'varL', 'varR', 'betaL_odd', 'betaL_even', 'betaR_odd', 'betaR_even'):
        p = f'{wd}/{n}.nii.gz'
        if not os.path.exists(f'{wd}/{n}_prep.nii.gz'):
            shutil.copy(p, f'{wd}/{n}_prep.nii.gz')
        img = nib.load(f'{wd}/{n}_prep.nii.gz'); v = np.asarray(img.dataobj, np.float32)
        v = np.where(mask & np.isfinite(v), v, 0).astype(np.float32)
        nib.save(nib.Nifti1Image(v, img.affine, img.header), p)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), mimg.affine), f'{wd}/mask.nii.gz')
    b = np.abs(np.asarray(nib.load(f'{wd}/betaL.nii.gz').dataobj, float))[mask]
    print(S, os.path.basename(wd), 'runs', len(runs), 'removed', int((mask0 & ~mask).sum()), 'of', int(mask0.sum()), 'voxels; max |betaL| now', round(float(b.max()), 1), flush=True)
