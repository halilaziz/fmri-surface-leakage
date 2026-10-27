"""Week 4: fill the StudyForrest part of week4_report.md from sf_summarize.py's outputs (tables are copied, not retyped).
Usage: python3 assemble_report.py REPORT_TEMPLATE SF_SUMMARY_DIR OUT_REPORT
The template holds @@SF@@ (replaced here); @@SUMMARY@@ is written by hand after reading the numbers.
"""
import sys, json, re
import numpy as np

TPL, SD, OUT = sys.argv[1:4]
sec = open(f'{SD}/sf_report_section.md').read().split('\n')
G = json.load(open(f'{SD}/sf_group.json'))


def block(start_pat, stop_pats):
    i = next(k for k, l in enumerate(sec) if l.startswith(start_pat))
    j = next((k for k in range(i + 1, len(sec)) if any(sec[k].startswith(p) for p in stop_pats)), len(sec))
    return sec[i:j]


out = ['### Pre-specified comparison (settings fixed from HCP), visual vertices', '']
vis = block('### visual vertices', ['### pericalcarine'])[1:]
out += [l for l in vis if l.strip()] + ['']
pooled = block('Pooled over pairs and subjects (pre-specified', ['### Exploratory'])
out += [l for l in pooled if l.strip()] + ['',
        'Covered-vertex and pericalcarine tables: `sf/sf_report_section.md`. Figure: `figW4-3_sf_prediction.png` (differences to '
        'standard + smoothing, one dot per subject).', '']

# same-resolution reference: 3T -> 7T at equal voxel size vs 3T -> finer 7T, best competitor, visual vertices
best = lambda p: min(G[p]['visual']['smoothing'][0], G[p]['visual']['resample_smoothing'][0]) if p in G else np.nan
rows = []
for r in ('r30', 'r20', 'r14'):
    same = f'3T-{r}->7T-{r}'
    finer = [p for p in G if p.startswith(f'3T-{r}->7T-') and float(p[-2:]) / 10 < float(r[1:]) / 10 and not p.endswith('r08')]
    if same in G:
        rows.append(f"| 3T {int(r[1:]) / 10:.1f} mm | {best(same):.3f} | " +
                    (', '.join(f"{p.split('->')[1].replace('7T-r', '7T ').replace('14', '1.4 mm').replace('20', '2.0 mm')}: {best(p):.3f}"
                               for p in sorted(finer)) or '–') + f" | {G[same]['visual']['ceiling_split_half'][0]:.3f} |")
out += ['### How much of the residual is resolution?', '',
        'Best competitor (smoothing or resample + smoothing), visual vertices: 3T source predicting the 7T session at the same voxel '
        'size vs predicting finer 7T sessions (the 0.8 mm target is left out: its own noise dominates).', '',
        '| 3T source | → 7T same voxel size | → finer 7T targets | target odd vs even (same-size target) |', '|---|---|---|---|'] + rows + ['']

# exploratory
curves = block('### Exploratory (not pre-specified): smoothing-width curves', ['### Exploratory (not pre-specified): every'])
retuned = block('### Exploratory (not pre-specified): every', ['Pooled over pairs and subjects (re-tuned'])
pooled2 = block('Pooled over pairs and subjects (re-tuned', ['### Registration'])
hdr = [l for l in retuned if l.startswith('| pair') or l.startswith('|---')]
body = [l for l in retuned if l.startswith('| ') and '| visual |' in l]
out += ['### Exploratory: settings re-tuned on StudyForrest (leave-one-subject-out), visual vertices', '',
        next(l for l in retuned if l.startswith('Smoothing FWHM')), ''] + hdr + body + [''] + [l for l in pooled2 if l.strip()] + ['',
        'Smoothing-width curves (each method at every setting, mean over subjects): `sf/sf_report_section.md`; figures '
        '`figW4-4_sf_fwhm_curves_exploratory.png`, `figW4-5_sf_retuned_exploratory.png`. Example maps: `figW4-6_sf_example_maps.png`.', '']
reg = block('### Registration', ['@@none@@'])
out += [l for l in reg] + ['']
txt = open(TPL).read().replace('@@SF@@', '\n'.join(out).rstrip() + '\n')
open(OUT, 'w').write(txt)
print('\n'.join(out))
