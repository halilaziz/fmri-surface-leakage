"""Week 15, analysis C: summarise the realistic-PSF leakage maps over subjects (mean over hemispheres, then over subjects).
Usage (local): python3 src/15_far_part_tests/leak_realistic_summary.py IN_DIR OUT_PREFIX
"""
import sys, json, glob
import numpy as np

J = [json.load(open(f)) for f in sorted(glob.glob(f'{sys.argv[1]}/*.week15_leak_realistic.json'))]
subs = [j['subject'] for j in J]
VAR = ['emul', 'native', 'native_pf78']
VLAB = {'emul': 'emulated kernel (Tukey window; as published)', 'native': 'native-acquisition kernel at that voxel size', 'native_pf78': 'native kernel with partial Fourier 7/8 (magnitude approximation)'}
KEYS = [('leak_frac_mean', 'mean far leak (%)', 100), ('frac_leak_frac_gt5', 'cortex with far leak > 5% (%)', 100), ('frac_leak_frac_gt10', '> 10% (%)', 100),
        ('gross_far_frac_mean', 'mean gross far weight (%)', 100), ('frac_gross_gt20', 'cortex with gross far weight > 20% (%)', 100),
        ('calc_leak_frac_mean', 'pericalcarine far leak (%)', 100), ('dilution_mean', 'dilution of a unit field', 1)]
res = {'subjects': subs, 'grids': {}}
L = [f'# Week 15 C: leakage of standard mapping at 2.4 / 3.2 mm for realistic native-acquisition PSFs ({len(subs)} development subjects)', '',
     'Area-weighted means per hemisphere, averaged over hemispheres and subjects (± SD over subjects). "Emulated": the kernel of the k-space',
     'emulation used throughout the paper (native 1.6 mm kernel times a Tukey window at the coarse Nyquist). "Native": the week-2 native PSF',
     'model evaluated for an acquisition at that voxel size (Tukey-tapered sinc in plane, T2* decay over the echo train along PE with the',
     'same echo spacing and FOV, erf slice profile, interpolation blur). "PF 7/8": the same with the PE band reduced to 7/8 (the magnitude',
     'PSF of a zero-filled partial-Fourier reconstruction). Far = other hemisphere or > 15 mm on the area-scaled sphere.',
     'Source: `src/15_far_part_tests/leak_realistic.py`; per-subject files in `leak_realistic/`.', '']
for g in ('2.4', '3.2'):
    res['grids'][g] = {}
    L += [f'## Emulated grid {g} mm', '', '| quantity | ' + ' | '.join(VLAB[v] for v in VAR) + ' |', '|---|---|---|---|']
    fw = {v: np.array([j['grids'][g][v]['psf_fwhm_mm_xyz'] for j in J]).mean(0) for v in VAR}
    L.append('| effective PSF FWHM x / y / z (mm) | ' + ' | '.join(' / '.join(f'{x:.2f}' for x in fw[v]) for v in VAR) + ' |')
    res['grids'][g]['psf_fwhm_mm_xyz'] = {v: fw[v].round(3).tolist() for v in VAR}
    for k, lab, scale in KEYS:
        cells = []
        for v in VAR:
            x = np.array([0.5 * (j['grids'][g][v]['L'][k] + j['grids'][g][v]['R'][k]) for j in J]) * scale
            res['grids'][g].setdefault(k, {})[v] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)))
            cells.append(f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if scale == 100 else f'{x.mean():.3f} ± {x.std(ddof=1):.3f}')
        L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    L.append('')
open(f'{sys.argv[2]}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{sys.argv[2]}.json', 'w'), indent=1)
print('\n'.join(L))
