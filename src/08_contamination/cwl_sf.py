"""Week 8, plan A, StudyForrest: contrast-weighted leak (CWL) at the coarse grid of the week-7 within-session emulation (one
subject, 7T session, factor; both directions). The coarse grid, its ribbon weights and its operator are rebuilt exactly as
week-7 sf_gate.py; response estimates (complex hemifield maps, real and imaginary parts as two maps) from the week-7 file:
  CWL-M  source half's native standard map (the other direction's target: zn[A] = 'BA_target', zn[B] = 'AB_target')
  CWL-P  source's coarse data upsampled in k-space, mapped at the native grid ('{d}_upsample')
The far-leak fraction is recomputed as a check against the week-7 'far_c'.
Usage (xcp_d container, TRUBA): python3 cwl_sf.py SUBJECT SESSION --factor F [--plan-sha1 SHA]
"""
import os, sys, json, time, argparse, shutil
import numpy as np, nibabel as nib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '06_noise_dose_response')); sys.path.insert(0, HERE)
from w6lib import *
from cwl_lib import sphere_frames, far_contamination, cwl, split_complex

GATE = f'{ROOT}/results/week7/sf_gate'
OUT = f'{ROOT}/results/week8/cwl_sf'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('subject'); ap.add_argument('session'); ap.add_argument('--factor', type=float, required=True)
    ap.add_argument('--plan-sha1', default='')
    a = ap.parse_args(); S, key, fac = a.subject, a.session, a.factor
    od = f'{OUT}/{S}'; os.makedirs(od, exist_ok=True)
    tmp = f'{od}/tmp_{key}_x{fac:g}_{os.getpid()}'; os.makedirs(tmp, exist_ok=True)
    T0 = time.time(); say = lambda *x: print(S, key, f'x{fac:g}', f'{time.time() - T0:6.0f}s', *x, flush=True)
    Z = np.load(f'{GATE}/{S}/{S}.{key}.x{fac:g}.w7_gate.npz')
    geo, v2r = load_anatomy(S); ses = load_session(S, key, v2r)
    sph = {h: nib.freesurfer.read_geometry(f'{SF}/freesurfer/{S}/surf/{fh}.sphere')[0] for h, fh in (('L', 'lh'), ('R', 'rh'))}
    U = unknowns(geo, [(ses['A_anat'], ses['shape'])])
    assert np.array_equal(U['idx'], Z['idx']), 'unknowns differ from the week-7 run'
    shape, wd, acq = ses['shape'], ses['wd'], ses['acq']
    mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float32)
    mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
    cs = coarse_shape(shape, fac)
    A_epi_c, A_anat_c = coarse_affine(ses['A_epi'], shape, cs), coarse_affine(ses['A_anat'], shape, cs)
    Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]
    Em = apply_axes(Ms, mrun); good = Em > 0.05 * np.percentile(Em, 99)
    mask_c = (apply_axes(Ms, mask.astype(np.float64)) > 0.9) & good.all(-1)
    tmpl = f'{tmp}/{key}_x{fac:g}_template.nii.gz'
    nib.save(nib.Nifti1Image(np.where(mask_c, Em.mean(-1), 0).astype(np.float32), A_epi_c), tmpl)
    W_c, _ = ribbon_weights(geo, U, A_epi_c, A_anat_c, tmpl, tmp, f'{key}_x{fac:g}')
    csp = voxel_sizes(dict(affine=A_anat_c))
    tabs_c, rad_c = coarse_tabs(acq, [acq['h'], acq['h'], 1.1 * acq['h']], csp)
    K_c = build_operator(U['P'], U['V'], U['Cc'], U['n'], dict(shape=cs, affine=A_anat_c), tabs_c, rad_c).tocsr()
    say('coarse grid rebuilt', cs)
    su, fs = sphere_frames(geo, sph, lambda V, F: vertex_areas_normals(V, F)[0])
    src_native = {'AB': Z['BA_target'], 'BA': Z['AB_target']}             # the source half's own native map
    Xs = {}
    for d in ('AB', 'BA'):
        Xs[f'M_{d}'] = split_complex(src_native[d]); Xs[f'P_{d}'] = split_complex(Z[f'{d}_upsample'])
    C, far_chk = far_contamination(W_c, K_c, U['idx'], U['nv']['L'], su, fs, Xs)
    dev = float(np.nanmax(np.abs(far_chk - Z['far_c'].astype(np.float64))))
    out = dict(idx=Z['idx'], plan_sha1=a.plan_sha1, far_check_max_abs_dev=dev)
    for d in ('AB', 'BA'):
        scored = Z[f'{d}_base'] & Z['visual']
        for t in 'MP':
            out[f'cwl{t}_{d}'] = cwl(C[f'{t}_{d}'], Xs[f'{t}_{d}'], scored).astype(np.float32)
    np.savez_compressed(f'{od}/{S}.{key}.x{fac:g}.cwl.npz', **out)
    vis = Z['AB_base'] & Z['visual']
    say('done; far-leak check max |dev|', f'{dev:.2e}', '; median CWL-M / CWL-P (visual, AB)',
        round(float(np.median(out['cwlM_AB'][vis])), 4), round(float(np.median(out['cwlP_AB'][vis])), 4))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
