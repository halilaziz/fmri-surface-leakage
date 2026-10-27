"""Week 5 shared code: StudyForrest subject/session geometry as in week 4 (sf_model.py), k-space emulation of coarser grids on
oblique EPI grids with exact per-axis operators (and their squares, for propagating voxel noise variances), and the model /
competitor settings grids.
"""
import os, sys, json, glob, time
import numpy as np, scipy.sparse as sp, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_misregistration_studyforrest'))
from w3lib import *
import sf_model as SFM

ROOT, SF, PREP = SFM.ROOT, SFM.SF, SFM.PREP
PRIOR_GRID = SFM.PRIOR_GRID                     # (range mm, s.d. scale) x 20, as in the week-4 exploratory grid
FW_GRID = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
HCP_FIXED = {1.5: (12.0, 0.125), 2.0: (24.0, 0.25)}   # week-3 choices for 1.6 -> 2.4 mm (x1.5) and 1.6 -> 3.2 mm (x2)
resid_real = SFM.resid_real


def load_anatomy(S):
    """FreeSurfer surfaces in scanner RAS of orig.mgz, aparc-based ROI / visual / pericalcarine labels (as week 4)."""
    FS = f'{SF}/freesurfer/{S}'
    orig = nib.load(f'{FS}/mri/orig.mgz'); v2r = orig.affine; tkr2scan = v2r @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    geo = {}
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        W, F = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.white'); Pv, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.pial')
        lab, _, names = nib.freesurfer.read_annot(f'{FS}/label/{fh}.aparc.annot')
        nm = np.array([names[l] if l >= 0 else b'unknown' for l in lab])
        W, Pv = SFM.hom(tkr2scan, W), SFM.hom(tkr2scan, Pv)
        geo[h] = dict(white=W, pial=Pv, mid=0.5 * (W + Pv), faces=F.astype(np.int64),
                      roi=~np.isin(nm, [b'unknown', b'corpuscallosum']), visual=np.isin(nm, SFM.VISUAL), calc=nm == b'pericalcarine')
    return geo, v2r


def load_session(S, key, v2r):
    wd = f'{PREP}/{S}/{key}'; field, res = key.split('-')
    mimg = nib.load(f'{wd}/session_mean.nii.gz')
    typ, M = SFM.read_lta(f'{wd}/bbr.lta')
    A_anat = v2r @ M if typ == 0 else M @ mimg.affine              # EPI voxel -> anatomical scanner RAS
    return dict(key=key, field=field, res=res, wd=wd, shape=tuple(mimg.shape[:3]), A_anat=A_anat, A_epi=mimg.affine,
                acq=SFM.acq_params(field, res, S), img=mimg)


def unknowns(geo, grids, margin=5.0):
    """Cortex whose midthickness lies inside any of the grids [(A_anat, shape), ...] (+ margin mm); unknown index set, FEM
    matrices of the unknowns' submeshes, column quadrature points restricted to the unknowns."""
    nv = {h: len(geo[h]['white']) for h in 'LR'}
    g = {h: dict(geo[h]) for h in 'LR'}
    inside = {h: np.zeros(nv[h], bool) for h in 'LR'}
    for A, shape in grids:
        Ai = np.linalg.inv(A); mg = margin / voxel_sizes(dict(affine=A))
        for h in 'LR':
            q = SFM.hom(Ai, g[h]['mid'])
            inside[h] |= ((q > -0.5 - mg) & (q < np.array(shape) - 0.5 + mg)).all(1)
    for h in 'LR':
        g[h]['roi'] = g[h]['roi'] & inside[h]
    av = analysis_vertices(g)
    idx = np.concatenate([np.flatnonzero(av['L']), nv['L'] + np.flatnonzero(av['R'])]); n = len(idx)
    loc = -np.ones(nv['L'] + nv['R'], int); loc[idx] = np.arange(n)
    Cr, Gr = {}, {}
    for h in 'LR':
        Vr, Fr, _ = roi_submesh(g[h]['mid'], g[h]['faces'], av[h]); Cr[h], Gr[h] = fem_matrices(Vr, Fr)
    P, V, C, Z, _ = subject_points(g)
    keep = loc[C] >= 0
    cat = lambda d: np.concatenate([d['L'], d['R']])[idx]
    return dict(idx=idx, n=n, nv=nv, nroi={h: int(av[h].sum()) for h in 'LR'}, Cr=Cr, Gr=Gr, P=P[keep], V=V[keep], Cc=loc[C[keep]],
                cat=cat, visual=cat({h: geo[h]['visual'] for h in 'LR'}), calc=cat({h: geo[h]['calc'] for h in 'LR'}))


def ribbon_weights(geo, U, A_epi, A_anat, template, od, tag):
    """Workbench ribbon-constrained weights (vertices x voxels, rows normalised) for the grid of `template` (a NIfTI whose
    affine is A_epi); surfaces are moved into the EPI world through the registration. Returns a function voxels -> unknowns."""
    a2e = A_epi @ np.linalg.inv(A_anat)
    shape = nib.load(template).shape[:3]
    W = {}
    for h in 'LR':
        for nm in ('white', 'pial', 'mid'):
            SFM.write_surf(f'{od}/{tag}.{h}.{nm}.surf.gii', SFM.hom(a2e, geo[h][nm]), geo[h]['faces'], h)
        txt = f'{od}/{tag}.{h}.w.txt'
        wb_ribbon_weights(template, f'{od}/{tag}.{h}.mid.surf.gii', f'{od}/{tag}.{h}.white.surf.gii', f'{od}/{tag}.{h}.pial.surf.gii', txt)
        W[h] = parse_weights_text(txt, U['nv'][h], shape).tocsr(); os.remove(txt)
        for nm in ('white', 'pial', 'mid'):
            os.remove(f'{od}/{tag}.{h}.{nm}.surf.gii')
    return W, (lambda y, W=W: U['cat']({h: W[h] @ y for h in 'LR'}))


# ----------------------------------------------------------------------------- emulation on (oblique) EPI grids
def coarse_shape(shape, factor):
    return tuple(int(np.floor(n / factor + 0.5)) for n in shape)


def coarse_affine(A, shape, cshape):
    """Same field of view and first voxel centre as the k-space crop (sample 0 stays at sample 0); works for oblique grids."""
    S = np.diag([shape[d] / cshape[d] for d in range(3)] + [1.0])
    return A @ S


def emul_matrix(n, n2, taper=EMUL_TAPER):
    """(n2 x n) operator of week 2's emulate() along one axis: k-space crop to n2 samples times the Tukey window."""
    X = np.fft.fft(np.eye(n), axis=0)
    X = np.fft.ifftshift(np.fft.fftshift(X, axes=0)[n // 2 - n2 // 2: n // 2 - n2 // 2 + n2], axes=0)
    f = np.abs(np.fft.fftfreq(n2)) / 0.5
    a = f; w = np.ones_like(a); edge = (a > 1 - taper) & (a <= 1)
    w[edge] = 0.5 * (1 + np.cos(np.pi * (a[edge] - (1 - taper)) / taper)); w[a > 1] = 0
    return np.real(np.fft.ifft(X * w[:, None], axis=0)) * (n2 / n)


def apply_axes(Ms, vol):
    """Separable operator: M_0 along axis 0, M_1 along axis 1, M_2 along axis 2 (extra trailing axes are carried along)."""
    out = vol
    for ax, M in enumerate(Ms):
        out = np.moveaxis(np.tensordot(M, out, axes=([1], [ax])), 0, ax)
    return out


def coarse_tabs(acq, native_spacing, cspacing, rad_native=SUPPORT_VOX['1.6'], rad=SUPPORT_VOX['2.4']):
    """Kernels of the emulated acquisition along the three EPI voxel axes: the native kernels exactly as used on the native
    grid (week 4 session_tabs: support (4, 4, 3) voxels) times the emulation window of the coarse grid."""
    ntr = [(rad_native[d] + 0.5) * native_spacing[d] for d in range(3)]
    tabs = [kernel_table(d, 1.0, acq=acq, coarse_mm=cspacing[d], native_trunc_mm=ntr[d], trunc_mm=(rad[d] + 0.5) * cspacing[d],
                         R_mm=(rad[d] + 1) * cspacing[d], rolloff=ROLLOFF['coarse']) for d in range(3)]
    return tabs, rad


def smoother_cache(Cr, Gr):
    cache = {}

    def get(fw):
        if fw not in cache:
            cache[fw] = surface_smooth(Cr, Gr, fw)
        return cache[fw]
    return get
