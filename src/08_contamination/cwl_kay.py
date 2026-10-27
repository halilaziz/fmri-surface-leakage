"""Week 8, plan A, ds002702: contrast-weighted leak (CWL) of standard mapping at the coarse grid of the week-7 within-session
emulation (one subject, one factor, both directions). The coarse grid, its Workbench ribbon weights and its operator are rebuilt
exactly as kay_emul.py; the response estimates come from the week-7 predictions:
  CWL-M  source half's native 0.8 mm standard map ('{d}_native')
  CWL-P  source's coarse data upsampled in k-space, mapped at 0.8 mm ('{d}_upsample')
The far-leak fraction is recomputed as a check against the week-7 'far_c'.
Usage (xcp_d container): python3 cwl_kay.py SUBJECT --factor F
"""
import os, sys, json, time, argparse, shutil, glob
import numpy as np, nibabel as nib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '07_ds002702_gate_group_maps')); sys.path.insert(0, HERE)
from kay_emul import *
import kay_emul as KE
from cwl_lib import sphere_frames, far_contamination, cwl

OUT = f'{KE.G_}/results/kay/cwl'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('--factor', type=float, required=True)
    ap.add_argument('--session', default='ring7T'); ap.add_argument('--plan-sha1', default='')
    a = ap.parse_args(); S, fac, ses = a.subject, a.factor, a.session
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{ses}_x{fac:g}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, ses, f'x{fac:g}', f'{time.time() - T0:6.0f}s', *x, flush=True)
    Z = np.load(f'{KE.OUT}/{S}/{S}.{ses}.x{fac:g}.kay_emul.npz')
    wd = f'{PREPD}/{S}/{ses}'
    geo, sph = load_anatomy_fs(f"{FSD}/subj{S.split('-')[1]}")
    mimg = nib.load(f'{wd}/session_mean.nii.gz'); A_epi = mimg.affine; shape = mimg.shape[:3]
    A_anat = np.loadtxt(f'{wd}/epi2anat_vox2ras.txt')
    js = json.load(open(sorted(glob.glob(f'{DATA}/{S}/ses-{ses}/func/*task-ring_run-01_bold.json'))[0]))
    acq = acq_from_json(js, '3T' if js['MagneticFieldStrength'] < 5 else '7T')
    U = unknowns(geo, [(A_anat, shape)])
    assert np.array_equal(U['idx'], Z['idx']), 'unknowns differ from the week-7 run'
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    cs = coarse_shape(shape, fac)
    A_epi_c, A_anat_c = coarse_affine(A_epi, shape, cs), coarse_affine(A_anat, shape, cs)
    Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]
    Em = apply_axes(Ms, mrun); good = Em > 0.05 * np.percentile(Em, 99)
    mask_c = (apply_axes(Ms, mask.astype(np.float64)) > 0.9) & good.all(-1)
    tmpl = f'{tmp}/{ses}_x{fac:g}_template.nii.gz'
    nib.save(nib.Nifti1Image(np.where(mask_c, Em.mean(-1), 0).astype(np.float32), A_epi_c), tmpl)
    W_c, _ = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, tmp, f'{ses}_x{fac:g}')
    csp = voxel_sizes(dict(affine=A_anat_c))
    tabs_c, rad_c = coarse_tabs(acq, [acq['h']] * 3, csp)
    K_c = build_operator(U['P'], U['V'], U['Cc'], U['n'], dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
    say('coarse grid rebuilt', cs)
    su, fs = sphere_frames(geo, sph, lambda V, F: SFM_area(V, F))
    Xs = {f'{t}_{d}': Z[f'{d}_{src}'].astype(np.float64) for d in ('AB', 'BA') for t, src in (('M', 'native'), ('P', 'upsample'))}
    C, far_chk = far_contamination(W_c, K_c, U['idx'], U['nv']['L'], su, fs, Xs)
    dev = float(np.nanmax(np.abs(far_chk - Z['far_c'].astype(np.float64))))
    out = dict(idx=Z['idx'], plan_sha1=a.plan_sha1, far_check_max_abs_dev=dev)
    for d in ('AB', 'BA'):
        scored = Z[f'{d}_base'] & Z['visual']
        for t in 'MP':
            out[f'cwl{t}_{d}'] = cwl(C[f'{t}_{d}'], Xs[f'{t}_{d}'], scored).astype(np.float32)
    np.savez_compressed(f'{od}/{S}.{ses}.x{fac:g}.cwl.npz', **out)
    vis = Z['AB_base'] & Z['visual']
    say('done; far-leak check max |dev|', f'{dev:.2e}', '; median CWL-M / CWL-P (visual, AB)',
        round(float(np.median(out['cwlM_AB'][vis])), 4), round(float(np.median(out['cwlP_AB'][vis])), 4))
    shutil.rmtree(tmp, ignore_errors=True)


def SFM_area(V, F):
    return vertex_areas_normals(V, F)[0]


if __name__ == '__main__':
    main()
