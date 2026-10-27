"""Registration QC for ds002702 sessions: EPI intensity just outside vs just inside the FreeSurfer white surface (+-1 mm along the
normal), inside the slab. With a correct subject and registration, gray matter (outside) is brighter than white matter at 7T GE-EPI,
consistently across vertices. Compared with the same registration shifted by 2 mm (a mis-registration reference) and with another
subject's FreeSurfer surfaces (a wrong-subject reference).
Usage (xcp_d container): [KAY_PREP=prep_dc] python3 kay_regcheck.py SUBJECT SESSION [OTHER_SUBJECT]
"""
import os, sys, numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates
G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
PREP = os.environ.get('KAY_PREP', 'prep')          # 'prep' (kay_prep.py) or 'prep_dc' (kay_undistort.py)


def contrast(S, ses, fs_subj, shift=np.zeros(3)):
    wd = f'{G}/results/kay/{PREP}/{S}/{ses}'
    img = nib.load(f'{wd}/session_mean.nii.gz'); I = np.asarray(img.dataobj, np.float32)
    A = np.loadtxt(f'{wd}/epi2anat_vox2ras.txt'); Ai = np.linalg.inv(A)
    fs = f'{G}/data/ds002702_freesurfer/{fs_subj}'
    orig = nib.load(f'{fs}/mri/orig.mgz'); t2s = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    out = []
    for h in ('lh', 'rh'):
        W, F = nib.freesurfer.read_geometry(f'{fs}/surf/{h}.white')
        W = W @ t2s[:3, :3].T + t2s[:3, 3] + shift
        fn = np.cross(W[F[:, 1]] - W[F[:, 0]], W[F[:, 2]] - W[F[:, 0]]); vn = np.zeros_like(W)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
        vals = []
        for d in (+1.0, -1.0):
            P = W + d * vn; q = P @ Ai[:3, :3].T + Ai[:3, 3]
            inside = ((q > 1) & (q < np.array(I.shape) - 2)).all(1)
            vals.append((map_coordinates(I, q.T, order=1, mode='constant', cval=np.nan), inside))
        ok = vals[0][1] & vals[1][1] & np.isfinite(vals[0][0]) & np.isfinite(vals[1][0]) & (vals[0][0] > 0) & (vals[1][0] > 0)
        c = (vals[0][0][ok] - vals[1][0][ok]) / (vals[0][0][ok] + vals[1][0][ok])
        out.append(c)
    c = np.concatenate(out)
    return dict(n=int(len(c)), median=float(np.median(c)), frac_positive=float(np.mean(c > 0)))


S, ses = sys.argv[1], sys.argv[2]
subj = f"subj{S.split('-')[1]}"
print(S, ses, 'registered            ', contrast(S, ses, subj))
for sh in ([2, 0, 0], [0, 2, 0], [0, 0, 2]):
    print(S, ses, f'shifted {sh} mm     ', contrast(S, ses, subj, np.array(sh, float)))
if len(sys.argv) > 3:
    print(S, ses, f'surfaces of {sys.argv[3]}', contrast(S, ses, sys.argv[3]))
