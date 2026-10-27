"""Week 6 diagnostic: spatial structure of the run-to-run noise in StudyForrest 7T betas, at the native grid and the emulated
coarse grids of week 5. Noise = each run's beta map minus the mean of its half (odd / even runs), scaled to unit variance per voxel
(so this measures correlation, not amplitude). Reports the autocorrelation at lags of 0-6 voxels along each axis and the radially
averaged power spectrum in physical frequency; saves them as JSON (and a figure).
Usage (xcp_d container): python3 noise_spectrum.py SUBJECT SESSION OUTDIR
"""
import os, sys, json
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '05_studyforrest_within_session'))
from w5lib import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S, key, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
wd = f'{PREP}/{S}/{key}'
bL = np.asarray(nib.load(f'{wd}/runbetaL.nii.gz').dataobj, np.float64)
bR = np.asarray(nib.load(f'{wd}/runbetaR.nii.gz').dataobj, np.float64)
mrun = np.asarray(nib.load(f'{wd}/runmean.nii.gz').dataobj, np.float64)
mask = np.asarray(nib.load(f'{wd}/mask.nii.gz').dataobj) > 0.5
img = nib.load(f'{wd}/session_mean.nii.gz'); A = img.affine; shape = img.shape[:3]
R = bL.shape[-1]; halves = [list(range(0, R, 2)), list(range(1, R, 2))]


def residuals(Y, m):
    """Unit-variance run residuals about each half's mean, inside mask m; list of 3-D arrays (L and R parts pooled)."""
    out = []
    for part in Y:
        for h in halves:
            r = part[..., h] - part[..., h].mean(-1, keepdims=True)
            sd = np.sqrt((r ** 2).sum(-1) / (len(h) - 1))
            ok = m & (sd > 0)
            for i in range(len(h)):
                out.append(np.where(ok, r[..., i] / np.where(ok, sd, 1), 0.0) * np.sqrt(len(h) / (len(h) - 1)))
    return out, ok


def analyse(res, m, vox):
    acf = {}
    for ax in range(3):
        vals = []
        for lag in range(0, 7):
            num = den = 0.0
            for r in res:
                a = r; b = np.roll(r, -lag, axis=ax)
                mm = m & np.roll(m, -lag, axis=ax)
                if lag:   # drop wrapped samples
                    sl = [slice(None)] * 3; sl[ax] = slice(r.shape[ax] - lag, None); mm = mm.copy(); mm[tuple(sl)] = False
                num += (a[mm] * b[mm]).sum(); den += mm.sum()
            vals.append(float(num / max(den, 1)))
        acf[f'axis{ax}'] = [v / vals[0] for v in vals]
    P = np.zeros(m.shape)
    for r in res:
        P += np.abs(np.fft.fftn(r)) ** 2
    P /= len(res) * m.sum()
    k = np.meshgrid(*[np.fft.fftfreq(n, d=vox[d]) for d, n in enumerate(m.shape)], indexing='ij')
    kr = np.sqrt(sum(kk ** 2 for kk in k))                       # cycles / mm
    edges = np.linspace(0, kr.max(), 25)
    idx = np.digitize(kr.ravel(), edges) - 1
    prof = [float(P.ravel()[idx == i].mean()) if (idx == i).any() else None for i in range(len(edges) - 1)]
    return dict(acf=acf, k_cycles_per_mm=[float(0.5 * (edges[i] + edges[i + 1])) for i in range(len(edges) - 1)], power=prof,
                white_level=1.0, low_k_power=float(P.ravel()[kr.ravel() < 0.05].mean()))


out = {'subject': S, 'session': key, 'grids': {}}
vox_n = voxel_sizes(dict(affine=A))
res, ok = residuals((bL, bR), mask)
out['grids']['native'] = dict(voxel_mm=vox_n.tolist(), **analyse(res, mask, vox_n))
for fac in (1.5, 2.0):
    cs = coarse_shape(shape, fac); Ms = [emul_matrix(shape[d], cs[d]) for d in range(3)]
    Em = apply_axes(Ms, mrun); good = Em > 0.05 * np.percentile(Em, 99)
    yc = [np.where(good, 100 * apply_axes(Ms, b * mrun / 100) / np.where(good, Em, 1), 0) for b in (bL * mask[..., None], bR * mask[..., None])]
    mask_c = (apply_axes(Ms, mask.astype(float)) > 0.9) & good.all(-1)
    rc, _ = residuals(yc, mask_c)
    vox_c = voxel_sizes(dict(affine=coarse_affine(A, shape, cs)))
    out['grids'][f'x{fac:g}'] = dict(voxel_mm=vox_c.tolist(), **analyse(rc, mask_c, vox_c))
json.dump(out, open(f'{OUT}/{S}.{key}.noise_spectrum.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for g, G in out['grids'].items():
    ax[0].plot(np.arange(7) * G['voxel_mm'][0], G['acf']['axis0'], 'o-', label=f"{g} ({G['voxel_mm'][0]:.1f} mm)")
    kk = np.array(G['k_cycles_per_mm']); pw = np.array([p if p is not None else np.nan for p in G['power']])
    ax[1].semilogy(kk, pw, 'o-', ms=3, label=g)
ax[0].set_xlabel('lag along axis 0 (mm)'); ax[0].set_ylabel('noise autocorrelation'); ax[0].legend(frameon=False); ax[0].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('spatial frequency (cycles/mm)'); ax[1].set_ylabel('noise power (white = 1)'); ax[1].legend(frameon=False)
fig.suptitle(f'{S} {key}: run-to-run noise (unit variance per voxel)'); plt.tight_layout(); plt.savefig(f'{OUT}/{S}.{key}.noise_spectrum.png', dpi=130)
for g, G in out['grids'].items():
    print(g, 'voxel', np.round(G['voxel_mm'], 2), 'ACF axis0', np.round(G['acf']['axis0'], 3), 'axis2', np.round(G['acf']['axis2'], 3),
          'low-k power', round(G['low_k_power'], 2))
