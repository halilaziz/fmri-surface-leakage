"""Week 7, ds002702 real cross-session test: residual of the 7T 0.8 mm target maps per pair and stage (visual vertices), with the
pre-specified settings (left) and with settings chosen on the other subject of the pair type (right).
Usage (local): python3 plot_kay_cross.py kay_cross_results.json kay_cross_cv_results.json OUT.png
"""
import os
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P, C = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rows = []                                        # (label, prespec record, cv record)
for src, name in (('ring3T', '3T 2.4 mm'), ('ringSE', 'SE 1.05 mm')):
    for stage in ('uncorrected', 'field-map corrected'):
        for k, r in sorted(C.items()):
            s_, st_, v_, S = k.split('|')
            if s_ == src and st_ == stage and v_ == 'visual':
                pk = f"{S}|{src}-ring7T ({stage})|visual"
                rows.append((f"{S}\n{name}\n{'corrected' if stage != 'uncorrected' else 'uncorr.'}", P[pk], r))
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))     # font scale for print-size figures (week 14; 1 = as developed)
plt.rcParams.update({'font.size': FS * 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
w = 0.2
cols = [('competitor', 'best competitor', '#9e9e9e'), ('model', 'model', '#1f77b4'), ('gated', 'leak-gated', '#ff7f0e'),
        ('split_half', "target's own odd vs even runs", '#2ca02c')]
for a, which, title in ((ax[0], 1, 'settings pre-specified from the within-session emulation'),
                        (ax[1], 2, 'settings chosen on the other subject of the pair type (second analysis)')):
    for i, (c, lab, col) in enumerate(cols):
        a.bar(np.arange(len(rows)) + (i - 1.5) * w, [r[which][c] for r in rows], w, label=lab, color=col)
    a.set_xticks(range(len(rows))); a.set_xticklabels([r[0] for r in rows], fontsize=FS * 7)
    a.set_title(title, fontsize=FS * 8)
ax[0].set_ylabel('residual of the 7T 0.8 mm maps (lower is better)'); ax[0].legend(frameon=False, fontsize=FS * 7, ncol=2, loc='upper left')
fig.suptitle('ds002702: a separately acquired session predicts the 7T GE 0.8 mm session (visual cortex, six ring maps)', fontsize=FS * 9)
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=150); plt.close()
