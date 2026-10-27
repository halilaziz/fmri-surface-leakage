"""Manuscript Figure 1 (schematic): a sulcus in cross-section with a 2.4 mm voxel grid; the averaging kernel of standard
ribbon mapping at one vertex (voxels that intersect the ribbon near the vertex, blurred by an effective PSF of 3.8 mm FWHM),
plotted along the sheet; its far part (> 15 mm along the surface) comes from the opposite bank, close in the volume. Right:
the far part biases the mapped value only when the two banks respond differently.
Usage (local): python3 fig1_concept.py OUT.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

H, FWHM, GAP, DEPTH, THICK = 2.4, 3.8, 4.0, 20.0, 2.5           # voxel, PSF, bank separation, sulcal depth, thickness (mm)
sig = FWHM / 2.3548
# midthickness curve of a sulcus: left bank down, round fundus, right bank up (arc length s)
n = 2000
t = np.linspace(0, 1, n)
left = np.c_[np.full(n // 3, -GAP / 2), np.linspace(DEPTH, 0, n // 3)]
ang = np.linspace(np.pi, 2 * np.pi, n // 3)
fund = np.c_[GAP / 2 * np.cos(ang), GAP / 2 * np.sin(ang)]
right = np.c_[np.full(n // 3, GAP / 2), np.linspace(0, DEPTH, n // 3)]
C = np.vstack([left, fund, right])
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
v = np.argmin(np.abs(C[:, 1] - 11.0) + 100 * (C[:, 0] > 0))     # vertex on the left bank, 11 mm above the fundus
sv = s[v]
# voxels of standard mapping at v: grid voxels whose centre lies within the ribbon around v (|dy| < 2 voxels on the left bank)
gx = np.arange(-12, 12 + 1e-9, H) + H / 2; gy = np.arange(-6, DEPTH + 6 + 1e-9, H) + H / 2
GX, GY = np.meshgrid(gx, gy)
dist_to_curve = np.min(np.linalg.norm(np.c_[GX.ravel(), GY.ravel()][:, None, :] - C[None, ::10, :], axis=2), axis=1).reshape(GX.shape)
in_ribbon = dist_to_curve < THICK / 2 + H / 2
near_v = (np.hypot(GX - C[v, 0], GY - C[v, 1]) < 1.6 * H) & in_ribbon & (GX < 0)
# readout of a unit field on the sheet: sum over the vertex's voxels of the PSF at each sheet point
pts = C
w = np.zeros(len(pts))
for x0, y0 in zip(GX[near_v], GY[near_v]):
    w += np.exp(-0.5 * ((pts[:, 0] - x0) ** 2 + (pts[:, 1] - y0) ** 2) / sig ** 2)
ds = np.r_[np.diff(s), 0]
w = w * ds; w /= w.sum()
geod = s - sv
far = np.abs(geod) > 15
lam = w[far].sum()

FS = float(os.environ.get('GEO_FONT_SCALE', '1'))     # font scale for print-size figures (week 14; 1 = as developed)
plt.rcParams.update({'font.size': FS * 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(13, 4.2))
a1 = fig.add_axes([0.03, 0.10, 0.26, 0.82]); a2 = fig.add_axes([0.36, 0.16, 0.30, 0.74]); a3 = fig.add_axes([0.73, 0.16, 0.25, 0.74])
# left: geometry
for x0, y0 in zip(GX.ravel(), GY.ravel()):
    a1.add_patch(Rectangle((x0 - H / 2, y0 - H / 2), H, H, fill=False, lw=0.3, color='0.75'))
for x0, y0 in zip(GX[near_v], GY[near_v]):
    a1.add_patch(Rectangle((x0 - H / 2, y0 - H / 2), H, H, color='#1f77b4', alpha=0.25, lw=0))
nrm = np.gradient(C, axis=0); nrm = np.c_[-nrm[:, 1], nrm[:, 0]]; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
for sgn in (-1, 1):
    a1.plot(*(C + sgn * THICK / 2 * nrm).T, color='0.35', lw=0.8)
sc = a1.scatter(C[::4, 0], C[::4, 1], c=np.where(far[::4], w[::4], np.nan), cmap='Oranges', s=6, vmin=0, vmax=w[far].max() * 1.2)
a1.scatter(*C[v], color='#1f77b4', s=40, zorder=5, label='vertex v')
th = np.linspace(0, 2 * np.pi, 100)
a1.plot(C[v, 0] + FWHM / 2 * np.cos(th) * 2, C[v, 1] + FWHM / 2 * np.sin(th) * 2, color='#1f77b4', lw=0.8, ls='--')
a1.annotate('opposite bank:\nclose in the volume,\n> 15 mm along the sheet', xy=(GAP / 2, C[v, 1]), xytext=(4.5, 15), fontsize=FS * 7,
            arrowprops=dict(arrowstyle='->', lw=0.6))
a1.set_xlim(-10, 10); a1.set_ylim(-5, DEPTH + 2); a1.set_aspect('equal'); a1.axis('off')
a1.set_title(f'Sulcus with {H} mm voxels; PSF {FWHM} mm FWHM', fontsize=FS * 8, loc='left')
# middle: readout along the sheet
a2.fill_between(geod, 0, w / ds.mean(), where=~far, color='#1f77b4', alpha=0.4, label='near (own) cortex')
a2.fill_between(geod, 0, w / ds.mean(), where=far, color='#ff7f0e', alpha=0.7, label=f'far cortex: far leak λ = {100 * lam:.0f}%')
a2.axvline(-15, color='k', lw=0.6, ls=':'); a2.axvline(15, color='k', lw=0.6, ls=':')
a2.set_xlabel('distance from v along the sheet (mm)'); a2.set_ylabel('readout weight of standard mapping (a.u.)')
a2.set_xlim(geod.min(), geod.max()); a2.legend(frameon=False, fontsize=FS * 7)
a2.set_title('Averaging kernel of standard mapping at v', fontsize=FS * 8, loc='left')
# right: harm only if the banks differ
cases = [('banks respond alike\n(e.g. eccentricity rings)', 1.0, 1.0), ('banks differ\n(e.g. polar angle across\nthe calcarine)', 1.0, -1.0)]
for i, (lab, xl, xr) in enumerate(cases):
    resp = np.where(C[:, 0] < 0, xl, xr); resp[(C[:, 1] < 0)] = 0.5 * (xl + xr)
    mapped = (w * resp).sum()
    a3.bar(i - 0.17, xl, 0.3, color='0.6', label='true response at v' if i == 0 else None)
    a3.bar(i + 0.17, mapped, 0.3, color='#1f77b4', label='standard-mapped value' if i == 0 else None)
    a3.text(i + 0.17, mapped + 0.03, f'{mapped:.2f}', ha='center', fontsize=FS * 7)
a3.set_xticks([0, 1]); a3.set_xticklabels([c[0] for c in cases], fontsize=FS * 7); a3.set_ylim(0, 1.25); a3.legend(frameon=False, fontsize=FS * 7)
a3.set_title('Leak harms only where the far cortex differs', fontsize=FS * 8, loc='left')
plt.savefig(sys.argv[1], dpi=200); plt.close()
print(f'far leak at v: {lam:.3f}')
