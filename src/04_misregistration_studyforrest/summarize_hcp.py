"""Week 4, HCP part: figures and report tables from evaluate_hcp.py's week4_hcp_eval.json.
Usage: python3 summarize_hcp.py EVAL_JSON OUTDIR
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
R = json.load(open(EV)); subs = R['subjects']; GR = ('2.4', '3.2')
STR = ('coherent', 'clean16_leak5', 'clean16_leak10', 'calc')
SLAB = {'coherent': 'all coherent', 'clean16_leak5': 'clean at 1.6, leak > 5%', 'clean16_leak10': 'clean at 1.6, leak > 10%',
        'calc': 'pericalcarine'}
MCOL = {'standard': '#999999', 'smoothing': '#8FAADC', 'upsample_smoothing': '#55A868', 'model': '#C44E52'}
MLAB = {'standard': 'standard mapping', 'smoothing': 'standard + smoothing (10 mm)',
        'upsample_smoothing': 'upsampling + 1.6 mm mapping + smoothing (8 mm)', 'model': 'model-based'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def sc(g, lab, s):
    return np.array([R['shift'][S][g].get(lab, {}).get(s, np.nan) if R['shift'][S][g].get(lab) else np.nan for S in subs], float)


def lab_at(m, d):
    return m if d == 0 else f'{m}_shift{d:g}'


def mse(x):
    x = x[np.isfinite(x)]
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan, len(x))


def paired(a, b):
    """a - b per subject: mean, sd, n, count(a < b)."""
    d = a - b; ok = np.isfinite(d); d = d[ok]
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=int(ok.sum()), a_better=int((d < 0).sum())) if ok.sum() > 1 else None


tables = {}
# ---------------- figure 1: misregistration and depth
fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
for gi, g in enumerate(GR):
    for si, s in enumerate(('coherent', 'clean16_leak10')):
        a = ax[gi, si]
        for m in ('standard', 'smoothing', 'upsample_smoothing', 'model'):
            ys = [mse(sc(g, lab_at(m, d), s)) for d in (0, 0.5, 1)]
            if all(np.isfinite(y[0]) for y in ys):
                a.errorbar([0, 0.5, 1], [y[0] for y in ys], yerr=[y[1] for y in ys], color=MCOL[m], marker='o', ms=4, capsize=2,
                           label=MLAB[m])
            elif np.isfinite(ys[0][0]):
                a.axhline(ys[0][0], color=MCOL[m], lw=1, ls=':', label=MLAB[m] + ' (nominal only)')
        a.set_xticks([0, 0.5, 1]); a.set_xlabel('registration error of the coarse data (mm)')
        a.set_ylabel('cross-run residual vs 1.6 mm map'); a.set_title(f'{g} mm, {SLAB[s]}', fontsize=9)
        if gi == 0 and si == 0:
            a.legend(frameon=False, fontsize=7)
    # depth panel: b1/b0 per subject and the score change of the depth-resolved model
    a = ax[gi, 2]
    for k, vs in enumerate(('coherent', 'pericalcarine')):
        v = np.array([R['depth'][S][g][vs]['b1_over_b0'] for S in subs if g in R['depth'][S]])
        a.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)), v, s=12, color='#8172B2')
        a.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color='k')
    dd = [sc(g, 'model_depth_full', s) - sc(g, 'model', s) for s in STR]
    a2 = a.twinx()
    DX = {'coherent': 'Δ\nall', 'clean16_leak5': 'Δ\nleak>5%', 'clean16_leak10': 'Δ\nleak>10%', 'calc': 'Δ\ncalc.'}
    for k, (s, d) in enumerate(zip(STR, dd)):
        a2.scatter(np.full(len(d), 2.6 + 0.8 * k) + np.random.default_rng(10 + k).uniform(-0.12, 0.12, len(d)), d, s=12, color='#DD8452')
        a2.plot([2.6 + 0.8 * k - 0.25, 2.6 + 0.8 * k + 0.25], [np.nanmean(d)] * 2, color='k')
    a2.axhline(0, color='#DD8452', lw=0.8, ls=':'); a2.set_ylabel('residual change, depth-resolved − b0-only model', color='#DD8452')
    a.set_xticks([0, 1] + [2.6 + 0.8 * k for k in range(len(STR))])
    a.set_xticklabels(['b1/b0\ncoherent', 'b1/b0\npericalc.'] + [DX[s] for s in STR], fontsize=7)
    a.set_ylabel('depth-gradient / column-mean amplitude', color='#8172B2'); a.set_title(f'{g} mm, white-to-pial gradient (b1)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW4-1_misregistration_depth.png', dpi=150); plt.close()

# ---------------- figure 2: uncertainty calibration
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for gi, g in enumerate(GR):
    for k, (lab, key) in enumerate((('model (MC-corrected)', 'model:noise_dominated'), ('standard', 'standard:noise_dominated'))):
        fld = 'mean_ratio_mc_corrected'
        v = np.array([R['calibration'][S][g][key][fld] for S in subs])
        x = gi * 2.6 + k
        ax[0].scatter(np.full(len(v), x) + np.random.default_rng(gi * 7 + k).uniform(-0.15, 0.15, len(v)), v, s=12,
                      color=MCOL['model' if k == 0 else 'standard'])
        ax[0].plot([x - 0.3, x + 0.3], [v.mean()] * 2, color='k')
ax[0].axhline(1, color='k', lw=0.8, ls=':'); ax[0].set_yscale('log')
ax[0].set_yticks([1, 1.5, 2, 3, 4, 6, 8]); ax[0].set_yticklabels(['1', '1.5', '2', '3', '4', '6', '8']); ax[0].minorticks_off()
ax[0].set_xticks([0, 1, 2.6, 3.6]); ax[0].set_xticklabels(['model\n2.4 mm', 'standard\n2.4 mm', 'model\n3.2 mm', 'standard\n3.2 mm'])
ax[0].set_ylabel('observed / predicted run-to-run variance'); ax[0].set_title('noise-dominated vertices (coherence < 0.1)', fontsize=9)
for ai, vs in enumerate(('all', 'coherent')):
    a = ax[1 + ai]; nom = np.array([0.5, 0.8, 0.95])
    for gi, g in enumerate(GR):
        cov = np.array([[R['calibration'][S][g][f'sim_posterior:{vs}'][f'coverage_{p:g}'] for p in nom] for S in subs])
        a.errorbar(nom + (gi - 0.5) * 0.012, cov.mean(0), yerr=cov.std(0), marker='o', ms=4, capsize=2, label=f'{g} mm',
                   color=('#C44E52', '#8C2D2D')[gi])
    a.plot([0.4, 1], [0.4, 1], 'k:', lw=0.8); a.set_xlim(0.4, 1); a.set_ylim(0, 1)
    a.set_xlabel('nominal coverage of the posterior interval'); a.set_ylabel('observed coverage of the known truth')
    a.set_title(f'week-3 simulation, {vs} vertices', fontsize=9); a.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/figW4-2_uncertainty.png', dpi=150); plt.close()

# ---------------- tables
f3 = lambda t: f'{t[0]:.3f} ± {t[1]:.3f}'
T = ['## HCP (emulated 2.4 / 3.2 mm, 15 test subjects, week-3 settings fixed)', '']
T += ['### Misregistration: cross-run residual (mean ± SE over subjects); the same rigid error is applied to every method', '']
for g in GR:
    T += [f'**{g} mm**', '', '| method | error | ' + ' | '.join(SLAB[s] for s in STR) + ' |', '|---|---|' + '---|' * len(STR)]
    for m in ('standard', 'smoothing', 'upsample_smoothing', 'model'):
        for d in (0, 0.5, 1):
            v = [mse(sc(g, lab_at(m, d), s)) for s in STR]
            if np.isfinite(v[0][0]):
                T.append(f'| {MLAB[m]} | {d:g} mm | ' + ' | '.join(f3(x) for x in v) + ' |')
    T.append('')
    tables[f'shift_{g}'] = {}
    rows = ['| comparison | ' + ' | '.join(SLAB[s] for s in STR) + ' |', '|---|' + '---|' * len(STR)]
    for d in (0, 0.5, 1):
        for comp in ('upsample_smoothing', 'smoothing'):
            pr = {s: paired(sc(g, lab_at('model', d), s), sc(g, lab_at(comp, d), s)) for s in STR}
            if all(pr[s] for s in STR):
                tables[f'shift_{g}'][f'model_minus_{comp}_at_{d:g}mm'] = pr
                rows.append(f'| model − {comp.replace("_", " ")}, both at {d:g} mm | ' +
                            ' | '.join(f"{pr[s]['mean']:+.4f} ± {pr[s]['sd']:.4f} ({pr[s]['a_better']}/{pr[s]['n']})" for s in STR) + ' |')
    for d in (0.5, 1):
        pr = {s: paired(sc(g, lab_at('model', d), s), sc(g, 'upsample_smoothing', s)) for s in STR}
        if all(pr[s] for s in STR):
            tables[f'shift_{g}'][f'model_at_{d:g}mm_minus_upsample_smoothing_nominal'] = pr
            rows.append(f'| model at {d:g} mm − upsampling + smoothing without error | ' +
                        ' | '.join(f"{pr[s]['mean']:+.4f} ± {pr[s]['sd']:.4f} ({pr[s]['a_better']}/{pr[s]['n']})" for s in STR) + ' |')
    T += ['Paired differences (mean ± SD over subjects; in brackets: subjects where the model is better):', ''] + rows + ['']
T += ['### Depth component (b1 = white-to-pial gradient, orthogonal to the column mean)', '',
      '| grid | b1/b0 coherent | fraction positive | b1/b0 pericalcarine | fraction positive | ' +
      ' | '.join(f'Δ residual {SLAB[s]}' for s in STR) + ' |', '|---|---|---|---|---|' + '---|' * len(STR)]
for g in GR:
    dg = R['group'][f'depth_{g}']
    pr = {s: paired(sc(g, 'model_depth_full', s), sc(g, 'model', s)) for s in STR}
    tables[f'depth_{g}'] = dict(group=dg, full_minus_nominal=pr)
    T.append(f"| {g} mm | {dg['coherent']['b1_over_b0'][0]:.3f} ± {dg['coherent']['b1_over_b0'][1]:.3f} | {dg['coherent']['frac_positive'][0]:.2f} | "
             f"{dg['pericalcarine']['b1_over_b0'][0]:.3f} ± {dg['pericalcarine']['b1_over_b0'][1]:.3f} | {dg['pericalcarine']['frac_positive'][0]:.2f} | " +
             ' | '.join(f"{pr[s]['mean']:+.4f} ± {pr[s]['sd']:.4f} ({pr[s]['a_better']}/{pr[s]['n']})" for s in STR) + ' |')
T += ['', '### Uncertainty calibration', '',
      '| grid | quantity | model | standard mapping |', '|---|---|---|---|']
for g in GR:
    cg = R['group'][f'calibration_{g}']
    for vs in ('noise_dominated', 'coherent'):
        T.append(f"| {g} mm | observed/predicted run-to-run variance, {vs.replace('_', '-')} (mean ratio; model MC-corrected) | "
                 f"{f3(cg[f'model:{vs}']['mean_ratio_mc_corrected'])} | {f3(cg[f'standard:{vs}']['mean_ratio'])} |")
        T.append(f"| {g} mm | fraction beyond the 95% bound, {vs.replace('_', '-')} (nominal 0.05) | "
                 f"{f3(cg[f'model:{vs}']['frac_beyond_95'])} | {f3(cg[f'standard:{vs}']['frac_beyond_95'])} |")
        k2 = 'frac_beyond_95_after_other_hemisphere_scale'
        if vs == 'noise_dominated' and k2 in cg[f'model:{vs}']:
            T.append(f"| {g} mm | same, after one variance inflation factor estimated on the other hemisphere | "
                     f"{f3(cg[f'model:{vs}'][k2])} | {f3(cg[f'standard:{vs}'][k2])} |")
    for vs in ('all', 'coherent'):
        c = cg[f'sim_posterior:{vs}']
        T.append(f"| {g} mm | simulation: posterior coverage at 50 / 80 / 95%, {vs} vertices | "
                 f"{c['coverage_0.5'][0]:.2f} / {c['coverage_0.8'][0]:.2f} / {c['coverage_0.95'][0]:.2f} | – |")
T += ['', '### Averaging-kernel spread vs week-1 resolvable wavelength (Spearman over the 24 week-3 kernel vertices per subject)', '',
      '| grid | model kernels | standard-mapping kernels | wavelength range (mm) |', '|---|---|---|---|']
for g in GR:
    th = R['group'][f'theory_{g}']
    lr = np.array([R['theory'][S][g]['lam_range_mm'] for S in subs if g in R['theory'][S]])
    T.append(f"| {g} mm | ρ = {th['rho_model_spread'][0]:+.3f} ± {th['rho_model_spread'][1]:.3f} (n = {th['rho_model_spread'][2]}) | "
             f"ρ = {th['rho_std_spread'][0]:+.3f} ± {th['rho_std_spread'][1]:.3f} (n = {th['rho_std_spread'][2]}) | "
             f"{lr[:, 0].mean():.2f}–{lr[:, 1].mean():.2f} |")
open(f'{OUT}/hcp_report_section.md', 'w').write('\n'.join(T) + '\n')
json.dump(tables, open(f'{OUT}/week4_hcp_tables.json', 'w'), indent=1, default=float)
print('\n'.join(T))
