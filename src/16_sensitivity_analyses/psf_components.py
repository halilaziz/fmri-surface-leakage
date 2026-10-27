"""Week 16, B (part 2): the FWHM contribution of each component of the native 1.6 mm PSF model, per axis, with the provenance of
each parameter, and the emulated kernels at 2.4 / 3.2 mm for reference. Local; no data.
Usage: python3 src/16_sensitivity_analyses/psf_components.py results/week16/psf_components.md
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import kernel_table, ACQ7T, SUPPORT_VOX, ROLLOFF, EMUL_TAPER


def fwhm(t, v):
    return 2 * float(t[np.argmax(v < v[0] / 2)])


rows = []
base = dict(ACQ7T)
# in-plane axes: sinc only -> + Tukey taper -> (+ T2* decay along PE) -> + interpolation blur
for axis, name in ((0, 'x (frequency encoding)'), (1, 'y (phase encoding)')):
    steps = [('sinc of the voxel (box in k-space)', dict(base, taper=1e-6, t2s_ms=1e9, sigma_interp=0.0)),
             (f"+ Tukey taper {base['taper']} (modelling assumption; no raw filter in the protocol)", dict(base, t2s_ms=1e9, sigma_interp=0.0))]
    if axis == base['pe_axis']:
        steps.append((f"+ T2* decay over the echo train (effective echo spacing {base['esp_eff_ms']} ms, FOV {base['fov_pe_mm']} mm, T2* {base['t2s_ms']} ms)", dict(base, sigma_interp=0.0)))
    steps.append((f"+ interpolation blur, σ = {base['sigma_interp']} mm (one-step resampling of the HCP pipeline; assumption)", dict(base)))
    for lab, acq in steps:
        t, v = kernel_table(axis, 1.0, acq=acq, R_mm=8.0)
        rows.append((name, lab, fwhm(t, v)))
# slice axis: erf-edged slice profile -> + Tukey roll-off -> + interpolation blur
for lab, acq in ((f"slice profile: box of the slice thickness with erf edge {base['slice_edge']} voxel (assumption), band-limited to 1.2× Nyquist", dict(base, sigma_interp=0.0)),
                 (f"+ interpolation blur, σ = {base['sigma_interp']} mm", dict(base))):
    t, v = kernel_table(2, 1.0, acq=acq, R_mm=8.0)
    rows.append(('z (slice)', lab, fwhm(t, v)))
L = ['# Week 16 B: components of the native 1.6 mm PSF model (FWHM in mm, MNI/ACPC scale 1.0)', '',
     f"Parameters: voxel {base['h']} mm; effective echo spacing {base['esp_eff_ms']} ms and FOV {base['fov_pe_mm']} mm along PE (HCP 7T protocol; iPAT 2 folded",
     f"into the effective spacing); T2* {base['t2s_ms']} ms (grey matter at 7T, literature value); in-plane Tukey taper {base['taper']}, slice edge {base['slice_edge']}",
     f"voxel and interpolation blur σ {base['sigma_interp']} mm are modelling assumptions. Partial Fourier (7/8 in the HCP 7T protocol) is not in the native model.", '',
     '| axis | components included | FWHM (mm) | voxels |', '|---|---|---|---|']
for name, lab, f in rows:
    L.append(f'| {name} | {lab} | {f:.2f} | {f / base["h"]:.2f} |')
L += ['', '## Emulated coarse kernels (native model times the Tukey-0.5 window at the coarse Nyquist)', '', '| grid | axis | FWHM (mm) | voxels |', '|---|---|---|---|']
for g in ('2.4', '3.2'):
    dv = float(g); rad = SUPPORT_VOX[g]
    for axis, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        ntr = (SUPPORT_VOX['1.6'][axis] + 0.5) * base['h']
        t, v = kernel_table(axis, 1.0, acq=base, coarse_mm=dv, native_trunc_mm=ntr, trunc_mm=(rad[axis] + 0.5) * dv, R_mm=(rad[axis] + 1) * dv, rolloff=ROLLOFF['coarse'])
        L.append(f'| {g} mm | {name} | {fwhm(t, v):.2f} | {fwhm(t, v) / dv:.2f} |')
L += ['', 'Reading: a bare sinc has a main lobe of 1.21 voxels; the taper, the T2* decay along PE and the interpolation blur bring the native kernel to',
      f'{rows[2][2] / base["h"]:.2f} (x), {rows[6][2] / base["h"]:.2f} (y) and {rows[-1][2] / base["h"]:.2f} (z) voxels. The emulated coarse kernels inherit this width (1.6–1.7 voxels).']
open(sys.argv[1], 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
