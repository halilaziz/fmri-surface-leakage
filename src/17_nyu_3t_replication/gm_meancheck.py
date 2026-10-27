"""Check (descriptive): per subject, area-weighted mean far leak in subject space (midthickness areas, analysis vertices) against
the same map resampled to fsaverage (3-NN on sphere.reg) and weighted by fsaverage midthickness areas, as the group map is.
Read-only on GEO_CONST inputs; prints one row per subject and the means (results/week17/groupmaps_meancheck.md).
Usage (fMRIPrep container on hebb): python3 gm_meancheck.py [N_SUBJECTS]"""
import os, sys, json, glob
import numpy as np, nibabel as nib
from scipy.spatial import cKDTree
G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
FSD = f'{G}/data/ds003787/derivatives/freesurfer'
MAPS = f'{G}/results/nyu/emul_gate'
FSAVG = os.environ.get('FSAVG', '/opt/freesurfer/subjects/fsaverage')
unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
def areas(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    o = np.zeros(len(V)); np.add.at(o, F.ravel(), np.repeat(a / 3, 3)); return o
fs = {}
for h, fh in (('L', 'lh'), ('R', 'rh')):
    W, F = nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.white'); P = nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.pial')[0]
    fs[h] = dict(sph=unit(nib.freesurfer.read_geometry(f'{FSAVG}/surf/{fh}.sphere.reg')[0]), area=areas(0.5 * (W + P), F))
rows = []
for f in sorted(glob.glob(f'{MAPS}/*.nyu_maps.npz'))[:int(sys.argv[1]) if len(sys.argv) > 1 else 1000]:
    S = os.path.basename(f).split('.')[0]; M = np.load(f); idx = M['idx']
    geo = {}
    for h, fh in (('L', 'lh'), ('R', 'rh')):
        W, F = nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.white'); P = nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.pial')[0]
        geo[h] = dict(n=len(W), sph=unit(nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{fh}.sphere.reg')[0]), area=areas(0.5 * (W + P), F))
    nL = geo['L']['n']; out = [S]
    for key in ('far_n', 'far_3.0', 'far_4.0'):
        v = M[key].astype(float); sub_num = sub_den = fs_num = fs_den = 0.0
        for h in 'LR':
            full = np.full(geo[h]['n'], np.nan); sel = (idx < nL) if h == 'L' else (idx >= nL); full[idx[sel] - (0 if h == 'L' else nL)] = v[sel]
            ok = np.isfinite(full); sub_num += (geo[h]['area'][ok] * full[ok]).sum(); sub_den += geo[h]['area'][ok].sum()
            d, j = cKDTree(geo[h]['sph'][ok]).query(fs[h]['sph'], k=3); w = 1 / np.maximum(d, 1e-6); r = (w * full[ok][j]).sum(1) / w.sum(1)
            near = d[:, 0] < 0.02; fs_num += (fs[h]['area'][near] * r[near]).sum(); fs_den += fs[h]['area'][near].sum()
        out += [sub_num / sub_den, fs_num / fs_den]
    rows.append(out); print(S, ' '.join(f'{x:.4f}' for x in out[1:]), flush=True)
R = np.array([r[1:] for r in rows])
print('MEAN subject-space / fsaverage-resampled:', ' | '.join(f'{k}: {R[:, 2*i].mean():.4f} / {R[:, 2*i+1].mean():.4f}' for i, k in enumerate(('2.0', '3.0', '4.0'))))
