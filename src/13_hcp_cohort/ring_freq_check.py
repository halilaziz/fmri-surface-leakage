"""Week 13, step 0: the temporal frequency of the HCP ring runs (RETEXP, RETCON), measured from the data, so the ring analysis
uses the right DFT bin. For each of the six retinotopy runs of one subject: mean power spectrum over pericalcarine cortical voxels
(voxels carrying ribbon weight to pericalcarine vertices), over the full 300-frame run and over the week-2 window (frames 22-277).
The wedge runs are the positive control (period 32 s: 9.375 cycles per 300 s; bin 8 of the 256-frame window).
Usage (xcp_d container): python3 ring_freq_check.py SUBJECT OUT.json
"""
import os, sys, json
import numpy as np, nibabel as nib, scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_forward_operator'))
from w2lib import HCP, CONCAT   # data root and the concatenated run name

S, OUT = sys.argv[1:3]
W2 = f'{os.environ.get("GEO_ROOT", "/arf/scratch/" + os.environ["USER"] + "/geo_model_res")}/results/week2/subjects/{S}'
RUNS = {'RETCCW': 0, 'RETCW': 300, 'RETEXP': 600, 'RETCON': 900, 'RETBAR1': 1200, 'RETBAR2': 1500}
img = nib.load(f'{HCP}/{S}/MNINonLinear/Results/{CONCAT}/{CONCAT}_hp2000_clean.nii.gz')
# pericalcarine voxels: ribbon weights (vertices x voxels) of pericalcarine vertices, both hemispheres
vox = np.zeros(int(np.prod(img.shape[:3])), bool)
for h in 'LR':
    W = sp.load_npz(f'{W2}/hcpW_1.6_{h}_all.npz').tocsr()
    L = nib.load(f'{HCP}/{S}/MNINonLinear/Native/{S}.{h}.aparc.native.label.gii')
    key = [k for k, v in L.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
    calc = np.asarray(L.darrays[0].data) == key
    vox |= np.asarray(W[calc].sum(0)).ravel() > 0
vi = np.flatnonzero(vox)
res = {'subject': S, 'n_voxels': int(len(vi))}
for run, o in RUNS.items():
    Y = np.asarray(img.dataobj[..., o:o + 300], dtype=np.float32).reshape(-1, 300)[vi].astype(np.float64)
    r = {}
    for name, (a, b) in (('full300', (0, 300)), ('win256', (22, 278))):
        X = Y[:, a:b]; X = X - X.mean(1, keepdims=True)
        P = (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(0); P[0] = 0
        n = b - a; k = int(np.argmax(P[1:n // 2])) + 1
        top = np.argsort(P[1:n // 2])[::-1][:5] + 1
        r[name] = dict(peak_bin=k, peak_cycles_per_s=k / n, peak_period_s=n / k, peak_share=float(P[k] / P.sum()),
                       top5_bins=[int(t) for t in top], top5_periods_s=[round(n / t, 2) for t in top])
    res[run] = r
    print(run, 'full300 peak period', round(r['full300']['peak_period_s'], 2), 's | win256 peak bin', r['win256']['peak_bin'],
          'period', round(r['win256']['peak_period_s'], 2), 's', flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
