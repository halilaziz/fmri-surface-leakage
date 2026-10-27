"""Week 15 figure (manuscript Fig. 9): the three reviewer-requested tests.
(A) far-part ablation: the model's gain over the competitor by far-leak bin with the full, near-only and renormalised-near
operators, 2.4 and 3.2 mm (15 development subjects, mean ± SE); (B) real coarser StudyForrest sessions: residual of the coarse
session's map predicted through the coarse operator with the near part only, near + far, renormalised near and near + shuffled
far, by far-leak stratum (7 subjects, pooled over the six session pairs) and for the primary pair 1.4 → 3.0 mm; (C) leakage of
standard mapping at 2.4 / 3.2 mm for the emulated kernel and two realistic native-acquisition kernels (20 development subjects).
Usage (local): python3 src/15_far_part_tests/fig_week15.py OUT_PNG [--leak results/week15/leak_realistic_summary.json]
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('out_png')
ap.add_argument('--eval', default='results/week15/ablation/eval'); ap.add_argument('--sf', default='results/week15/sf_operator_summary.json')
ap.add_argument('--leak', default='results/week15/leak_realistic_summary.json')
a = ap.parse_args()
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))
plt.rcParams.update({'font.size': FS * 9, 'axes.spines.top': False, 'axes.spines.right': False})
BINK = ['bin_0-0.02', 'bin_0.02-0.05', 'bin_0.05-0.1', 'bin_0.1-0.2', 'bin_0.2-1.01']; BL = ['0–2%', '2–5%', '5–10%', '10–20%', '> 20%']
fig = plt.figure(figsize=(14, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, 1.05], wspace=0.42)
# ---- (A) ablation
J = [json.load(open(f)) for f in sorted(glob.glob(f'{a.eval}/*.json'))]
for gi, g in enumerate(('2.4', '3.2')):
    ax = fig.add_subplot(gs[0, gi])
    for m, lab, st, col in (('model', 'full operator', 'o-', '#C44E52'), ('near', 'far part removed', 's--', '#4C72B0'), ('nearn', 'far part removed, rows renormalised', '^:', '#55A868')):
        G = np.array([[(j['grids'][g][m][k] - j['grids'][g]['upsample_smoothing'][k]) if (j['grids'][g][m].get(k) is not None and j['grids'][g]['upsample_smoothing'].get(k) is not None) else np.nan for k in BINK] for j in J])
        mu = np.nanmean(G, 0); se = np.nanstd(G, 0, ddof=1) / np.sqrt(np.isfinite(G).sum(0))
        ax.errorbar(np.arange(5) + {'model': -0.08, 'near': 0, 'nearn': 0.08}[m], mu, yerr=se, fmt=st, color=col, ms=4, capsize=2, label=lab)
    ax.axhline(0, color='k', lw=0.8, ls=':'); ax.set_xticks(range(5)); ax.set_xticklabels(BL, fontsize=FS * 7.5)
    ax.set_title(('A  Far-part ablation, ' if gi == 0 else 'A (cont.)  ') + f'emulated {g} mm', loc='left', fontsize=FS * 9)
    ax.set_xlabel('far leak of standard mapping at the coarse grid', fontsize=FS * 8)
    if gi == 0:
        ax.set_ylabel('model − competitor (mean ± SE, 15 subjects)'); ax.legend(frameon=False, fontsize=FS * 7, loc='lower left')
# ---- (B) StudyForrest
S = json.load(open(a.sf))['pairs']
ax = fig.add_subplot(gs[0, 2])
sets = ['visual_lf_lt2', 'visual_lf_gt5', 'visual_lf_gt10', 'visual_lf_gt20']; slab = ['< 2%', '> 5%', '> 10%', '> 20%']
x = np.arange(len(sets)); w = 0.2
for k, (m, lab, col) in enumerate((('near', 'near part only', '#4C72B0'), ('full', 'near + far', '#C44E52'), ('renorm', 'near, renormalised', '#55A868'), ('shuffled', 'near + shuffled far', '#999999'))):
    v = [S['pooled over pairs'][s]['means'][m] for s in sets]; vp = [S['r14->r30'][s]['means'][m] for s in sets]
    ax.bar(x + (k - 1.5) * w, v, w, color=col, label=lab)
    ax.plot(x + (k - 1.5) * w, vp, 'k_', ms=7, mew=1.2)
ax.set_xticks(x); ax.set_xticklabels([f'far leak {s}' if i == 0 else s for i, s in enumerate(slab)], fontsize=FS * 7.5)
ax.set_ylabel('residual of the coarse session\'s map'); ax.set_ylim(0.25, 0.5)
ax.set_title('B  Real coarser sessions (StudyForrest 7T)', loc='left', fontsize=FS * 9)
ax.set_xlabel('visual cortex, by predicted far leak at the coarse session\n(bars: six pairs pooled; dashes: 1.4 → 3.0 mm)', fontsize=FS * 8)
ax.legend(frameon=False, fontsize=FS * 7, loc='upper left')
# ---- (C) realistic PSFs
L = json.load(open(a.leak))['grids']
ax = fig.add_subplot(gs[0, 3])
var = ['emul', 'native', 'native_pf78']; vlab = ['emulated\n(as used)', 'native', 'native,\nPF 7/8']
x = np.arange(3); w = 0.38
for gi, (g, col) in enumerate((('2.4', '#DD8452'), ('3.2', '#C44E52'))):
    v = [L[g]['frac_leak_frac_gt5'][k]['mean'] for k in var]; e = [L[g]['frac_leak_frac_gt5'][k]['sd'] for k in var]
    ax.bar(x + (gi - 0.5) * w, v, w, yerr=e, capsize=2, color=col, label=f'{g} mm')
    m = [L[g]['leak_frac_mean'][k]['mean'] for k in var]
    for xi, mi, vi, ei in zip(x + (gi - 0.5) * w, m, v, e):
        ax.text(xi, vi + ei + 0.6, f'{mi:.1f}', ha='center', va='bottom', fontsize=FS * 7, color='0.3')
ax.set_xticks(x); ax.set_xticklabels(vlab, fontsize=FS * 7.5); ax.set_ylabel('cortex with far leak > 5% (%; mean ± SD)')
ax.set_title('C  Leak for realistic PSFs', loc='left', fontsize=FS * 9); ax.legend(frameon=False, fontsize=FS * 7.5)
ax.set_xlabel('kernel at the coarse grid\n(numbers: mean far leak, %)', fontsize=FS * 8)
fig.savefig(a.out_png, dpi=170, bbox_inches='tight'); plt.close(fig); print('wrote', a.out_png)
