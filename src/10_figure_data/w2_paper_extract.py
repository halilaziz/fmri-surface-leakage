"""Paper figures: the two week-2 inputs that exist only per subject on TRUBA, reduced to small files for the local redraw
(manuscript/make_panels.py). Same selections as src/02_forward_operator/summarize.py:
  - distribution over cortex of the per-vertex far-leak fraction, pooled over subjects and hemispheres (figW2-1, middle panel),
    stored as the fraction of vertices at or above each x on the plotted log grid;
  - the example subject's medial-occipital polar-angle maps on the very inflated surface (figW2-3, top row), stored as the
    projected vertex coordinates, the faces and the per-face phase (NaN where not coherent).
Usage (xcp_d container): python3 w2_paper_extract.py WEEK2_SUBJECTS_DIR WEEK1_RESULTS_DIR OUTDIR
"""
import os, sys, json
import numpy as np
import nibabel as nib

SUBJ, W1, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
COMBOS = ['1.6_gv', '1.6_all', '2.4_all', '3.2_all']


def load_maps(p):
    return {da.meta['Name']: np.asarray(da.data, np.float64) for da in nib.load(p).darrays}


subs = sorted(d for d in os.listdir(SUBJ) if os.path.exists(f'{SUBJ}/{d}/{d}.week2_summary.json'))
print('subjects:', len(subs))

# ---------------- distribution over cortex (curve of figW2-1: x_i against 1 - i/n over the sorted pooled values)
pool = {c: [] for c in COMBOS}
for S in subs:
    for h in 'LR':
        m = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_leakage.func.gii')
        roi = m['dilution_1.6_all'] != 0
        for c in COMBOS:
            g, tag = c.split('_')
            pool[c].append(m[f'leak_fraction_{g}_{tag}'][roi])
xg = np.logspace(-1, np.log10(60), 600)
D = {'subjects': len(subs), 'hemispheres': 2 * len(subs), 'x_percent': xg.round(6).tolist()}
for c in COMBOS:
    x = np.sort(np.concatenate(pool[c])) * 100
    D[c] = {'n_vertices': int(len(x)), 'frac_ge': (1 - np.searchsorted(x, xg, side='left') / len(x)).round(6).tolist()}
json.dump(D, open(f'{OUT}/w2_leak_distribution.json', 'w'), indent=1)

# ---------------- example polar-angle maps (first subject, left hemisphere, medial occipital cap projected on the y-z plane)
S = subs[0]; h = 'L'
gi = nib.load(f'{W1}/maps/_summary/{S}.{h}.very_inflated.surf.gii')
Vi = gi.agg_data('NIFTI_INTENT_POINTSET'); Fi = gi.agg_data('NIFTI_INTENT_TRIANGLE')
roi = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_leakage.func.gii')['dilution_1.6_all'] != 0
mw = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_wedge.func.gii')
fn = np.cross(Vi[Fi[:, 1]] - Vi[Fi[:, 0]], Vi[Fi[:, 2]] - Vi[Fi[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
occ = Vi[:, 1] < np.percentile(Vi[:, 1], 35)
Fm = Fi[occ[Fi].all(1) & ((1 if h == 'L' else -1) * fn[:, 0] > 0.2)]
used, Fr = np.unique(Fm, return_inverse=True)
out = dict(y=Vi[used, 1].astype(np.float32), z=Vi[used, 2].astype(np.float32), faces=Fr.reshape(Fm.shape).astype(np.int32))
for c in ('1.6_gv', '2.4_all', '3.2_all'):
    vals = np.where(roi & (mw[f'coherence_{c}'] > 0.25), mw[f'theta_{c}'], np.nan)
    with np.errstate(invalid='ignore'):
        out[f'face_phase_{c}'] = np.angle(np.nanmean(np.exp(1j * vals[Fm]), axis=1)).astype(np.float32)
    print(c, 'coherent faces:', int(np.isfinite(out[f'face_phase_{c}']).sum()), 'of', len(Fm))
np.savez_compressed(f'{OUT}/w2_wedge_example.npz', **out)
