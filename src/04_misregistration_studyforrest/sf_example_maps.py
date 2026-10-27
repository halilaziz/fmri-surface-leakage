"""Week 4, StudyForrest: example maps for one subject and pair (occipital pole, flattened sphere).
Left-hemifield betas on the right hemisphere and right-hemifield betas on the left hemisphere: target session's standard map, the
source's standard map, standard + smoothing and the model (HCP settings), each scaled onto the target as in the score.
Usage (xcp_d container): python3 sf_example_maps.py SUBJECT PAIR OUTPNG     e.g. sub-18 '7T-r30->7T-r14' fig.png
"""
import os, sys
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

S, PAIR, OUTPNG = sys.argv[1:4]
ZOOM = 40.0                                  # degrees of the sphere around the pericalcarine centre
SF = os.environ.get('GEO_ROOT', '/path/to/geo_root') + '/data/studyforrest'
MD = os.environ.get('GEO_ROOT', '/path/to/geo_root') + f'/results/week4/sf/model/{S}'
D = np.load(f'{MD}/{S}.sf_predictions.npz')
src, tgt = PAIR.split('->')
idx = D['idx']; FS = f'{SF}/freesurfer/{S}'
geo = {}
for h, fh in (('L', 'lh'), ('R', 'rh')):
    W, F = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.white'); P, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.pial')
    geo[h] = dict(mid=0.5 * (W + P), faces=F)
nL = len(geo['L']['mid'])


def full(v):
    out = np.full(nL + len(geo['R']['mid']), np.nan, complex if np.iscomplexobj(v) else float); out[idx] = v
    return out[:nL], out[nL:]


# flat view of the occipital pole: azimuthal equidistant projection of the FreeSurfer sphere around the pericalcarine centroid
flat = {}
calc_full = full(D['calc'].astype(float))
for h in 'LR':
    fh = {'L': 'lh', 'R': 'rh'}[h]
    sph, _ = nib.freesurfer.read_geometry(f'{FS}/surf/{fh}.sphere'); u = sph / np.linalg.norm(sph, axis=1, keepdims=True)
    cm = np.nan_to_num(calc_full[0 if h == 'L' else 1]) > 0.5
    c0 = u[cm].mean(0); c0 /= np.linalg.norm(c0)
    e1 = np.array([0, 0, 1.0]) - c0[2] * c0; e1 /= np.linalg.norm(e1); e2 = np.cross(c0, e1)
    th = np.arccos(np.clip(u @ c0, -1, 1))
    phi = np.arctan2(u @ e2, u @ e1)
    flat[h] = dict(xy=np.c_[th * np.sin(phi), th * np.cos(phi)] * 180 / np.pi, keep=th < np.radians(ZOOM * 1.5))
cov = D[f'covered_{src}'] & D[f'covered_{tgt}']; vis = D['visual']
maps = {'target': D[f'z_{tgt}'], 'source standard': D[f'z_{src}'], 'standard + smoothing (10 mm)': D[f'{PAIR}|smoothing'],
        'model (HCP prior)': D[f'{PAIR}|model']}
titles = list(maps)
fig, ax = plt.subplots(2, len(titles), figsize=(3.3 * len(titles), 6.6))
for row, (h, part, lab) in enumerate((('R', np.real, 'left-hemifield betas, right hemisphere'),
                                      ('L', np.imag, 'right-hemifield betas, left hemisphere'))):
    hm = (idx >= nL) if h == 'R' else (idx < nL)
    sel = cov & vis & hm
    z = part(maps['target'])
    lim = np.nanpercentile(np.abs(z[sel]), 98)
    F = geo[h]['faces']; xy = flat[h]['xy']
    Fm = F[flat[h]['keep'][F].all(1)]
    for k, t in enumerate(titles):
        p = part(maps[t])
        c = 1.0 if t == 'target' else float((p[sel] @ z[sel]) / max(p[sel] @ p[sel], 1e-30))
        vv = full(np.where(cov, c * p, np.nan))[0 if h == 'L' else 1]
        with np.errstate(all='ignore'):
            fv = np.nanmean(vv[Fm], 1)
        a = ax[row, k]
        a.tripcolor(Triangulation(xy[:, 0], xy[:, 1], Fm), facecolors=np.zeros(len(Fm)), cmap='Greys', vmin=-3, vmax=1)
        ok = np.isfinite(fv)
        tpc = a.tripcolor(Triangulation(xy[:, 0], xy[:, 1], Fm, mask=~ok), facecolors=np.where(ok, fv, 0), cmap='RdBu_r', vmin=-lim, vmax=lim)
        a.set_aspect('equal'); a.axis('off'); a.set_xlim(-ZOOM, ZOOM); a.set_ylim(-ZOOM, ZOOM)
        a.set_title(t if row == 0 else '', fontsize=9)
    ax[row, 0].text(-0.05, 0.5, lab, transform=ax[row, 0].transAxes, fontsize=8, rotation=90, va='center', ha='right')
    cb = fig.colorbar(tpc, ax=ax[row, :].tolist(), shrink=0.7, pad=0.01); cb.set_label('beta (%), predictions scaled to target', fontsize=8)
fig.suptitle(f'{S}, {PAIR.replace("->", " → ")}: occipital pole (flattened sphere, ±40° around the pericalcarine centre); '
             'grey = not covered by both sessions', fontsize=9)
plt.savefig(OUTPNG, dpi=150, bbox_inches='tight'); plt.close()
print('saved', OUTPNG)
