"""Week 16 figure (manuscript Fig. 10): the last four reviewer-requested analyses.
(A) PSF-scale scan: cross-run residual relative to the modelled kernel (α = 1) against the stretch factor α, per tuning subject and
mean, both grids; dotted lines mark the best competitor's level. (B) Far criterion: mean net far leak of standard mapping at far
thresholds 10 / 15 / 20 mm and for the other hemisphere alone (20 development subjects, mean ± SD). (C) True geodesic against the
sphere proxy: net far leak under the geodesic (graph, calibrated) and sphere criteria, and the share of cortex whose > 5% label
changes (5 tuning subjects). (D) H3 with re-tuned half-run settings: Δ_clean(half) − Δ_clean(full) at the full-run settings and
re-tuned (15 development subjects, mean and bootstrap 95% interval). (E) Correlated-noise likelihood: correlated − white residual
by stratum (15 development subjects, mean and bootstrap 95% interval). Panels whose result files are missing are left blank.
Usage (local): python3 src/16_sensitivity_analyses/fig_week16.py OUT_PNG
"""
import os, sys, json, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_png = sys.argv[1]; R = 'results/week16'
FS = float(os.environ.get('GEO_FONT_SCALE', '1'))
plt.rcParams.update({'font.size': FS * 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'1.6': '#55A868', '2.4': '#4C72B0', '3.2': '#C44E52'}
CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
fig = plt.figure(figsize=(17, 4.3))
gs = fig.add_gridspec(1, 5, width_ratios=[1.15, 0.95, 1.0, 0.9, 1.1], wspace=0.5)
have = lambda f: os.path.exists(f)

# ---- (A) PSF-scale scan
ax = fig.add_subplot(gs[0, 0])
J = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/psf_scan/eval/*.variant_eval.json'))]
if J:
    AL = [0.7, 0.85, 1.0, 1.15, 1.3]
    for g in ('2.4', '3.2'):
        base = CHOSEN[g]
        D = np.array([[j['grids'][g][base + ('' if al == 1.0 else f'_psf{al:g}')]['coherent'] - j['grids'][g][base]['coherent'] for al in AL] for j in J])
        for row in D:
            ax.plot(AL, row, '-', color=COL[g], lw=0.6, alpha=0.35)
        ax.plot(AL, D.mean(0), 'o-', color=COL[g], lw=2, ms=5, label=f'{g} mm, all axes')
        for ax_, mk in (('y', 's'), ('z', '^')):
            Dy = np.array([[j['grids'][g][base + f'_psf{al:g}{ax_}']['coherent'] - j['grids'][g][base]['coherent'] for al in (0.85, 1.15)] for j in J]).mean(0)
            ax.plot([0.85, 1.15], Dy, mk, color=COL[g], mfc='white', ms=5)
        comp = np.mean([j['grids'][g]['upsmooth8']['coherent'] - j['grids'][g][base]['coherent'] for j in J])
        ax.axhline(comp, color=COL[g], lw=0.8, ls=':'); ax.text(1.31, comp, 'competitor', color=COL[g], fontsize=FS * 6.5, va='center', ha='left')
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(AL); ax.set_xticklabels(['0.7', '0.85', '1', '1.15', '1.3'])
    ax.set_xlabel('kernel stretch α (1 = modelled PSF)', fontsize=FS * 8); ax.set_ylabel('cross-run residual − residual at α = 1')
    ax.set_xlim(0.64, 1.42)
    ax.plot([], [], 's', color='0.3', mfc='white', ms=5, label='phase-encoding axis only')
    ax.plot([], [], '^', color='0.3', mfc='white', ms=5, label='slice axis only')
    ax.plot([], [], '-', color='0.5', lw=0.6, label='single subjects (all axes)')
    ax.legend(frameon=False, fontsize=FS * 6.5, loc='upper left')
ax.set_title('A  PSF-scale scan (5 tuning subjects)', loc='left', fontsize=FS * 9)

# ---- (B) thresholds
ax = fig.add_subplot(gs[0, 1])
if have(f'{R}/leak_thresholds_summary.json'):
    T = json.load(open(f'{R}/leak_thresholds_summary.json'))
    for g in ('1.6', '2.4', '3.2'):
        mu = [T['grids'][g][v]['leak_frac_mean']['mean'] for v in ('thr10', 'thr15', 'thr20')]; sd = [T['grids'][g][v]['leak_frac_mean']['sd'] for v in ('thr10', 'thr15', 'thr20')]
        ax.errorbar([10, 15, 20], mu, yerr=sd, fmt='o-', color=COL[g], ms=4, capsize=2, label=f'{g} mm')
        ax.errorbar([25.5], [T['grids'][g]['oh']['leak_frac_mean']['mean']], yerr=[T['grids'][g]['oh']['leak_frac_mean']['sd']], fmt='D', color=COL[g], mfc='white', ms=4, capsize=2)
    ax.set_xticks([10, 15, 20, 25.5]); ax.set_xticklabels(['10', '15', '20', 'other\nhemi.\nonly'], fontsize=FS * 7.5)
    ax.set_xlabel('far threshold along the sheet (mm)', fontsize=FS * 8); ax.set_ylabel(f'mean net far leak (%), {len(T["subjects"])} subjects')
    ax.axvline(15, color='k', lw=0.6, ls=':'); ax.legend(frameon=False, fontsize=FS * 7, loc='upper right'); ax.set_ylim(0, None)
ax.set_title('B  Far criterion: threshold', loc='left', fontsize=FS * 9)

# ---- (C) geodesic vs sphere
ax = fig.add_subplot(gs[0, 2])
if have(f'{R}/geodesic_summary.json'):
    G = json.load(open(f'{R}/geodesic_summary.json')); x = np.arange(3); w = 0.26
    for k, (key, lab, hatch) in enumerate((('leak_frac_mean_geo', 'geodesic (graph, 15 mm)', ''), ('leak_frac_mean_cal', f'geodesic, calibrated ({15 * G["graph_over_exact_median_mean"]:.1f} mm graph)', '//'), ('leak_frac_mean_sph', 'sphere proxy (paper)', '..'))):
        vals = [100 * G['grids'][g][key] for g in ('1.6', '2.4', '3.2')]
        ax.bar(x + (k - 1) * w, vals, w, color=[COL[g] for g in ('1.6', '2.4', '3.2')], alpha=0.9 - 0.25 * k, hatch=hatch, edgecolor='k', lw=0.4, label=lab)
    for i, g in enumerate(('1.6', '2.4', '3.2')):
        ax.text(x[i], max(100 * G['grids'][g][k_] for k_ in ('leak_frac_mean_geo', 'leak_frac_mean_cal', 'leak_frac_mean_sph')) + 0.15,
                f"label change\n{100 * G['grids'][g]['label_change_gt5']:.0f}% (> 5%)\nr = {G['grids'][g]['corr']:.2f}", ha='center', va='bottom', fontsize=FS * 6.3)
    ax.set_xticks(x); ax.set_xticklabels(['1.6 mm', '2.4 mm', '3.2 mm']); ax.set_ylabel(f'mean net far leak (%), {len(G["subjects"])} subjects')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.legend(frameon=False, fontsize=FS * 6.3, loc='upper left')
ax.set_title('C  Geodesic against sphere proxy', loc='left', fontsize=FS * 9)

# ---- (D) H3 re-tuned
ax = fig.add_subplot(gs[0, 3])
if have(f'{R}/h3_retuned.json'):
    H = json.load(open(f'{R}/h3_retuned.json'))
    rows = [(k for k in H['grids']['2.4'] if k.startswith('H3a (fixed')), (k for k in H['grids']['2.4'] if k.startswith('H3a′'))]
    kf = next(k for k in H['grids']['2.4'] if k.startswith('H3a (fixed')); kr = next(k for k in H['grids']['2.4'] if k.startswith('H3a′'))
    for gi, g in enumerate(('2.4', '3.2')):
        for k, (key, lab, mk) in enumerate(((kf, 'full-run settings kept', 'o'), (kr, 're-tuned for half runs', 's'))):
            st = H['grids'][g][key]; xpos = gi + (k - 0.5) * 0.3
            ax.errorbar([xpos], [st['mean']], yerr=[[st['mean'] - st['ci95'][0]], [st['ci95'][1] - st['mean']]], fmt=mk, color=COL[g], mfc='white' if k == 0 else COL[g], ms=6, capsize=3,
                        label=lab if gi == 0 else None)
            ax.text(xpos + 0.06, st['mean'], f"{st['n_pos']}/{st['n']}", ha='left', va='center', fontsize=FS * 6.5)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(['2.4 mm', '3.2 mm']); ax.set_xlim(-0.6, 1.6); ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.25)
    ax.set_ylabel('Δ_clean(half) − Δ_clean(full)  (> 0: gain shrinks)', fontsize=FS * 8); ax.legend(frameon=False, fontsize=FS * 6.5, loc='upper right')
ax.set_title('D  H3 with re-tuned settings', loc='left', fontsize=FS * 9)

# ---- (E) correlated noise: F (white settings, open) and F2 (own prior re-tuned, filled)
ax = fig.add_subplot(gs[0, 4])
if have(f'{R}/corr_summary.json'):
    C = json.load(open(f'{R}/corr_summary.json')); C2 = json.load(open(f'{R}/corr_retuned.json')) if have(f'{R}/corr_retuned.json') else None
    strata = ['coherent', 'calc', 'clean16_leak10', 'bin_0-0.02', 'bin_0.02-0.05', 'bin_0.05-0.1', 'bin_0.1-0.2', 'bin_0.2-1.01']
    slab = ['all coherent', 'pericalcarine', 'clean, leak > 10%', '0–2%', '2–5%', '5–10%', '10–20%', '> 20%']
    for gi, g in enumerate(('2.4', '3.2')):
        sets = [(C['grids'][g], 'o', 'white', f'{g} mm, at the white settings', -0.27)]
        if C2 and 'dev' in C2['grids'].get(g, {}):
            sets.append(({k: v for k, v in C2['grids'][g]['dev'].items()}, 's', COL[g], f'{g} mm, own prior re-tuned', -0.09))
        for D_, mk, mfc, lab, off in sets:
            mu = np.array([D_[s_]['diff'].get('mean', np.nan) for s_ in strata]); lo = np.array([D_[s_]['diff'].get('ci95', [np.nan, np.nan])[0] for s_ in strata]); hi = np.array([D_[s_]['diff'].get('ci95', [np.nan, np.nan])[1] for s_ in strata])
            xs = np.arange(len(strata)) + off + gi * 0.36
            ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], fmt=mk, color=COL[g], mfc=mfc, ms=4, capsize=2, label=lab)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(strata))); ax.set_xticklabels(slab, rotation=40, ha='right', fontsize=FS * 6.8)
    ax.set_ylabel('correlated − white residual (< 0: correlated better)', fontsize=FS * 8); ax.legend(frameon=False, fontsize=FS * 6.3, loc='upper left', ncol=1)
    ax.axvline(2.5, color='k', lw=0.5, ls=':'); ax.set_ylim(-0.005, ax.get_ylim()[1] * 1.3); ax.text(5, 0.006, 'by far-leak bin', ha='center', va='center', fontsize=FS * 7, color='0.3')
ax.set_title('E  Correlated-noise likelihood (15 subjects)', loc='left', fontsize=FS * 9)

fig.savefig(out_png, dpi=200, bbox_inches='tight'); print('wrote', out_png)
