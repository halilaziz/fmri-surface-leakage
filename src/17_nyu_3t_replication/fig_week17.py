"""Week 17 figure (manuscript Fig. 11): the external replication in the NYU Retinotopy Dataset (3T, 2.0 mm, sweeping bars).
(A) Overall gain of the model over the competitor (chosen on the tuning subjects) over responsive vertices, per test subject,
emulated 3.0 and 4.0 mm. (B) Gain by bin of far leak at the coarse grid (responsive vertices clean at 2.0 mm; mean ± SE over
subjects with the bin). (C) The same by contamination-weighted leak (CWL-M). (D) Far leak of standard mapping for the 3T
2.0 mm acquisition and its emulations: mean far leak and the share of cortex above 5% (mean ± SD over subjects).
With --gate (PLAN_NYU_GATE.md A; manuscript v2 Fig. 8): (E) the gated estimator (model where CWL-P > τ, competitor elsewhere)
against the competitor, by τ, test subjects (mean and bootstrap 95% interval); the τ chosen on the tuning subjects is ringed.
Usage (local): python3 src/17_nyu_3t_replication/fig_week17.py OUT_PNG [--test results/week17/nyu_test.json] [--gate results/week17/nyu_gate.json]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('out_png'); ap.add_argument('--test', default='results/week17/nyu_test.json'); ap.add_argument('--gate', default=''); ap.add_argument('--rows', type=int, default=1, help='2: panels in two rows (A B C / D E), for print')
a = ap.parse_args(); R = json.load(open(a.test))
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))
plt.rcParams.update({'font.size': FS * 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'2.0': '#55A868', '3.0': '#4C72B0', '4.0': '#C44E52'}
BL = ['0–2%', '2–5%', '5–10%', '10–20%', '> 20%']
GT = json.load(open(a.gate)) if a.gate else None
if a.rows == 2:                                   # print layout: A B C over D E (E twice as wide)
    fig = plt.figure(figsize=(13.5, 8.6)); gs = fig.add_gridspec(2, 6, wspace=1.1, hspace=0.42)
    SLOT = {'A': gs[0, 0:2], 'B': gs[0, 2:4], 'C': gs[0, 4:6], 'D': gs[1, 0:2], 'E': gs[1, 2:6]}
else:
    fig = plt.figure(figsize=(19.5 if GT else 15.5, 4.2)); gs = fig.add_gridspec(1, 5 if GT else 4, width_ratios=[0.9, 1.1, 1.1, 0.95] + ([1.1] if GT else []), wspace=0.42)
    SLOT = {k: gs[0, i] for i, k in enumerate('ABCDE'[:5 if GT else 4])}

ax = fig.add_subplot(SLOT['A'])
for i, g in enumerate(('3.0', '4.0')):
    st = R['grids'][g]['N1: Δ responsive']; v = np.array([x for x in st['values'] if x is not None], float)
    xs = i + np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(xs, v, s=12, color=COL[g], alpha=0.6, lw=0)
    ax.errorbar([i + 0.28], [st['mean']], yerr=[[st['mean'] - st['ci95'][0]], [st['ci95'][1] - st['mean']]], fmt='o', color='k', ms=5, capsize=3)
    ax.text(i + 0.14, v.max() + 0.002, f"{st['n_neg']}/{st['n']} model better", ha='center', va='bottom', fontsize=FS * 7.5)
    ymax = max(ymax, v.max()) if i else v.max()
ax.axhline(0, color='k', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['3.0 mm', '4.0 mm']); ax.set_xlim(-0.5, 1.6)
ax.set_ylim(ax.get_ylim()[0], ymax + 0.009)
ax.set_ylabel('model − competitor (residual; < 0: model better)', fontsize=FS * 8)
ax.set_title(f"A  Overall gain ({R['grids']['3.0']['N1: Δ responsive']['n']} test subjects)", loc='left', fontsize=FS * 9)

for k, (var, lab) in enumerate((('far', 'far leak'), ('cwlM', 'contamination (CWL-M)'))):
    ax = fig.add_subplot(SLOT['BC'[k]])
    for g in ('3.0', '4.0'):
        b = R['grids'][g][f'bins_{var}']; m = np.array(b['mean'], float); se = np.array(b['se'], float); n = np.array(b['n'])
        ok = n >= 3
        ax.errorbar(np.arange(5)[ok] + (-0.06 if g == '3.0' else 0.06), m[ok], yerr=se[ok], fmt='o-', color=COL[g], ms=4, capsize=2, label=f'{g} mm')
    ax.axhline(0, color='k', lw=0.8, ls=':'); ax.set_xticks(range(5)); ax.set_xticklabels(BL, fontsize=FS * 7.5)
    ax.set_xlabel(f'{lab} of standard mapping at the coarse grid' if a.rows == 1 else f'{lab} at the coarse grid', fontsize=FS * 8)
    if k == 0:
        ax.set_ylabel('model − competitor (mean ± SE)', fontsize=FS * 8); ax.legend(frameon=False, fontsize=FS * 7.5)
    ax.set_title(('B  Gain by far leak' if k == 0 else 'C  Gain by contamination'), loc='left', fontsize=FS * 9)

ax = fig.add_subplot(SLOT['D']); L = R['leak']; x = np.arange(3); w = 0.38
for j, (key, lab) in enumerate((('mean_far', 'mean far leak'), ('gt5', 'cortex > 5%'))):
    mu = [L[g][key]['mean'] for g in ('2.0', '3.0', '4.0')]; sd = [L[g][key]['sd'] for g in ('2.0', '3.0', '4.0')]
    ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=2, color=[COL[g] for g in ('2.0', '3.0', '4.0')], alpha=1.0 - 0.45 * j,
           edgecolor='k', lw=0.4, hatch='' if j == 0 else '//', label=lab)
ax.set_xticks(x); ax.set_xticklabels(['2.0 mm', '3.0 mm', '4.0 mm']); ax.set_ylabel('%'); ax.legend(frameon=False, fontsize=FS * 7.5)
ax.set_title('D  Far leak, 3T acquisition', loc='left', fontsize=FS * 9)
if GT:
    ax = fig.add_subplot(SLOT['E']); TAUS = ['0', '0.02', '0.05', '0.1', '0.2', 'inf']
    for g in ('3.0', '4.0'):
        r = GT['grids'][g]; d = 0.07 if g == '4.0' else -0.07
        m = np.array([np.nan if r['all_tau_test'][t]['vs_comp'].get('mean') is None else r['all_tau_test'][t]['vs_comp']['mean'] for t in TAUS], float)
        lo = np.array([r['all_tau_test'][t]['vs_comp']['ci95'][0] if 'ci95' in r['all_tau_test'][t]['vs_comp'] else np.nan for t in TAUS], float)
        hi = np.array([r['all_tau_test'][t]['vs_comp']['ci95'][1] if 'ci95' in r['all_tau_test'][t]['vs_comp'] else np.nan for t in TAUS], float)
        x = np.arange(len(TAUS)) + d
        ax.errorbar(x, m, yerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)], fmt='o-', color=COL[g], ms=4, capsize=2, label=f'{g} mm')
        k = TAUS.index(r['tau']); ax.plot([x[k]], [m[k]], 'o', ms=11, mfc='none', mec=COL[g], mew=1.6)
    ax.axhline(0, color='k', lw=0.8, ls=':'); ax.set_xticks(range(len(TAUS)))
    ax.set_xticklabels(['0\n(model)', '0.02', '0.05', '0.1', '0.2', '∞\n(competitor)'], fontsize=FS * 7.5)
    sh = {g: [100 * GT['grids'][g]['all_tau_test'][t]['share'] for t in TAUS] for g in ('3.0', '4.0')}
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.16 * (yl[1] - yl[0]))
    for j in range(len(TAUS)):
        ax.text(j, ax.get_ylim()[1], f"{sh['3.0'][j]:.0f}/{sh['4.0'][j]:.0f}", ha='center', va='top', fontsize=FS * 6.5, color='0.35')
    ax.text(len(TAUS) - 0.5, ax.get_ylim()[1] - 0.075 * (ax.get_ylim()[1] - ax.get_ylim()[0]), '% of vertices to the model (3.0/4.0 mm)',
            ha='right', va='top', fontsize=FS * 6.5, color='0.35')
    ax.set_xlabel('threshold τ on coarse-data contamination (CWL-P)', fontsize=FS * 8)
    ax.set_ylabel('gated − competitor (mean, 95% CI)', fontsize=FS * 8); ax.legend(frameon=False, fontsize=FS * 7.5, loc='lower right')
    ax.set_title('E  Model only where CWL-P > τ', loc='left', fontsize=FS * 9)
fig.savefig(a.out_png, dpi=200, bbox_inches='tight'); print('wrote', a.out_png)
