"""MVE week 3 summary: tables, figures and report from evaluate.py outputs.
Usage (xcp_d container): python3 summarize.py EVAL_DIR TEST_DIR OUTDIR
"""
import os, sys, json, subprocess
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3lib import *

EV, TD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
T = json.load(open(f'{EV}/week3_tuning.json')); R = json.load(open(f'{EV}/week3_test_scores.json'))
GR = [g for g in ('2.4', '3.2') if g in R['choice']]
MLAB = {'standard': 'standard mapping', 'upsample': 'k-space upsampling', 'smoothing': 'standard + surface smoothing',
        'upsample_smoothing': 'upsampling + 1.6 mm mapping + smoothing', 'model': 'model-based (this work)'}
MCOL = {'standard': '#999999', 'upsample': '#DDAA88', 'smoothing': '#8FAADC', 'upsample_smoothing': '#55A868', 'model': '#C44E52'}
SLAB = {'coherent': 'all coherent', 'clean16_leak5': 'clean at 1.6, leak > 5%', 'clean16_leak10': 'clean at 1.6, leak > 10%',
        'calc': 'pericalcarine'}
TICK = {'coherent': 'all\ncoherent', 'clean16_leak5': 'clean at 1.6,\nleak > 5%', 'clean16_leak10': 'clean at 1.6,\nleak > 10%',
        'calc': 'peri-\ncalcarine'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
subs = R['test_subjects']


def arr(g, m, s):
    return np.array([R['scores'][S][g][m].get(s, np.nan) for S in subs], float)


def ms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.mean(), x.std(), len(x)) if len(x) else (np.nan, np.nan, 0)


# ---------------- figure 1: tuning surfaces
fig, ax = plt.subplots(len(GR), 3, figsize=(13, 3.6 * len(GR)), squeeze=False)
for gi, g in enumerate(GR):
    tab = T['choice'][g]['table']
    for ci, crit in enumerate(('coherent', 'clean16_leak10')):
        mods = {k: v for k, v in tab[crit].items() if k.startswith('model_')}
        rhos = sorted({float(k.split('_')[1][1:]) for k in mods}); ss = sorted({float(k.split('_')[2][1:]) for k in mods})
        M = np.array([[mods.get(f'model_r{r:g}_s{s:g}', np.nan) for s in ss] for r in rhos])
        im = ax[gi, ci].imshow(M, cmap='viridis_r', aspect='auto')
        ax[gi, ci].set_xticks(range(len(ss))); ax[gi, ci].set_xticklabels([f'{s:g}' for s in ss])
        ax[gi, ci].set_yticks(range(len(rhos))); ax[gi, ci].set_yticklabels([f'{r:g}' for r in rhos])
        ax[gi, ci].set_xlabel('prior s.d. scale'); ax[gi, ci].set_ylabel('prior range (mm)')
        ax[gi, ci].set_title(f'{g} mm, model, tuning residual: {SLAB[crit]}', fontsize=8.5)
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax[gi, ci].text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7, color='w' if v > np.nanmean(M) else 'k')
    for crit, ls in (('coherent', '-'), ('clean16_leak10', '--')):
        sm = {float(k[6:]): v for k, v in T['choice'][g]['table'][crit].items() if k.startswith('smooth')}
        xs = sorted(sm); ax[gi, 2].plot(xs, [sm[x] for x in xs], 'o' + ls, color='#4C72B0', label=f'smoothing, {SLAB[crit]}')
        us = {float(k[8:]): v for k, v in T['choice'][g]['table'][crit].items() if k.startswith('upsmooth')}
        if us:
            xu = sorted(us); ax[gi, 2].plot(xu, [us[x] for x in xu], 's' + ls, color='#55A868', ms=4, label=f'upsampling + smoothing, {SLAB[crit]}')
        best = min(v for k, v in T['choice'][g]['table'][crit].items() if k.startswith('model_'))
        ax[gi, 2].axhline(best, color='#C44E52', ls=ls, lw=1, label=f'best model, {SLAB[crit]}')
        ax[gi, 2].axhline(T['choice'][g]['table'][crit]['standard'], color='#999999', ls=ls, lw=1, label=f'standard, {SLAB[crit]}')
    ax[gi, 2].set_xlabel('smoothing FWHM (mm)'); ax[gi, 2].set_ylabel('tuning residual'); ax[gi, 2].legend(fontsize=6.5, frameon=False)
    ax[gi, 2].set_title(f'{g} mm: competitors on the tuning subjects', fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{OUT}/figW3-1_tuning.png', dpi=150); plt.close()

# ---------------- figure 2: test-subject scores and paired comparison
fig, ax = plt.subplots(1, len(GR) + 1, figsize=(5.2 * len(GR) + 4.2, 3.9))
for gi, g in enumerate(GR):
    xs = np.arange(len(SLAB)); w = 0.16
    for mi, m in enumerate(MLAB):
        v = np.array([ms(arr(g, m, s))[:2] for s in SLAB])
        ax[gi].bar(xs + (mi - 2) * w, v[:, 0], w, yerr=v[:, 1], capsize=1.5, color=MCOL[m], label=MLAB[m])
    ax[gi].set_xticks(xs); ax[gi].set_xticklabels([TICK[s] for s in SLAB], fontsize=7.5)
    ax[gi].set_ylabel('cross-run residual vs 1.6 mm map'); ax[gi].set_title(f'{g} mm emulated, {len(subs)} test subjects')
    if gi == 0:
        ax[gi].legend(frameon=False, fontsize=7)
axp = ax[-1]
for gi, g in enumerate(GR):
    for si, s in enumerate(SLAB):
        d = arr(g, 'model', s) - np.fmin(arr(g, 'smoothing', s), arr(g, 'upsample_smoothing', s))
        x = gi * (len(SLAB) + 1) + si
        axp.scatter(np.full(len(d), x) + np.random.default_rng(si).uniform(-0.15, 0.15, len(d)), d, s=9, color='#C44E52')
        axp.plot([x - 0.3, x + 0.3], [np.nanmean(d)] * 2, 'k')
axp.axhline(0, color='k', lw=0.6)
axp.set_xticks([gi * (len(SLAB) + 1) + si for gi in range(len(GR)) for si in range(len(SLAB))])
axp.set_xticklabels([f'{g}\n{s.replace("clean16_", "")}' for g in GR for s in SLAB], fontsize=6.5)
axp.set_ylabel('model - best competitor (per subject)'); axp.set_title('Paired difference vs the better smoothing competitor (< 0: model better)', fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{OUT}/figW3-2_test_scores.png', dpi=150); plt.close()

# ---------------- figure 3: opposite-bank contrast, averaging kernels, simulation
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
xs = np.arange(len(GR) * 2); w = 0.16
for mi, m in enumerate(MLAB):
    vals = []
    for g in GR:
        for suf in ('', '_calc'):
            vals.append(ms([R['pairs'][S][f'{g}_{m}{suf}'] / R['pairs'][S][f'ref_1.6{suf}'] for S in subs])[:2])
    vals = np.array(vals)
    ax[0].bar(xs + (mi - 2) * w, vals[:, 0], w, yerr=vals[:, 1], capsize=1.5, color=MCOL[m], label=MLAB[m])
ax[0].axhline(1, color='k', lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels([f'{g} mm\n{t}' for g in GR for t in ('all pairs', 'pericalcarine')])
ax[0].set_ylabel('opposite-bank dissimilarity / 1.6 mm'); ax[0].set_title('Bank contrast kept (1 = as at 1.6 mm)'); ax[0].legend(frameon=False, fontsize=7)
for g, mk in zip(GR, ('o', 's')):
    rows = [r for S in subs for r in R['kernels'].get(S, {}).get(g, [])]
    if rows:
        ax[1].scatter([r['std_leak_fraction'] * 100 for r in rows], [r['model_leak_fraction'] * 100 for r in rows], s=10, marker=mk, label=f'{g} mm', alpha=0.7)
lim = ax[1].get_xlim(); ax[1].plot(lim, lim, 'k:', lw=0.8); ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('standard mapping: net leak from far cortex (%)'); ax[1].set_ylabel('model estimate: net leak (%)')
ax[1].set_title('Averaging kernels at sample vertices'); ax[1].legend(frameon=False)
xs = np.arange(len(SLAB)); w = 0.16
for mi, m in enumerate(MLAB):
    for gi, g in enumerate(GR):
        v = np.array([ms([R['simulation'][S][g][m][s] for S in subs if g in R['simulation'].get(S, {})])[:2] for s in SLAB])
        ax[2].bar(xs + gi * 0.42 + (mi - 2) * w / 2 - 0.21, v[:, 0], w / 2, color=MCOL[m], alpha=1 if gi == 0 else 0.55,
                  label=f'{MLAB[m]}' if gi == 0 else None)
ax[2].set_xticks(xs); ax[2].set_xticklabels([TICK[s] for s in SLAB], fontsize=7.5); ax[2].set_ylabel('residual vs known truth')
ax[2].set_title(f'Simulation, model exact (solid {GR[0]} mm' + (f', pale {GR[1]} mm)' if len(GR) > 1 else ')')); ax[2].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/figW3-3_banks_kernels_sim.png', dpi=150); plt.close()

# ---------------- figure 4: example polar-angle maps (first test subject, left hemisphere, medial occipital)
S = subs[0]; h = 'L'
E = np.load(f'{TD}/{S}/{S}.week3_estimates.npz'); idx = E['idx']
geo = hemi_geometry(S, h); nvL = len(geo['white'])       # L mesh vertices (global indices of L are < nvL)
vinfl = f'{OUT}/{S}.{h}.very_inflated.surf.gii'
if not os.path.exists(vinfl):
    mid = f'{HCP}/{S}/MNINonLinear/Native/{S}.{h}.midthickness.native.surf.gii'
    subprocess.run(['wb_command', '-surface-generate-inflated', mid, f'{OUT}/{S}.{h}.inflated.surf.gii', vinfl], check=True)
gi_ = nib.load(vinfl); Vi = gi_.agg_data('NIFTI_INTENT_POINTSET'); Fi = gi_.agg_data('NIFTI_INTENT_TRIANGLE')
nv = len(Vi)
delay = json.load(open(f'{W_ROOT}/results/week2/subjects/{S}/{S}.week2_summary.json'))['wedge']['delay_rad_mod_pi']
g = GR[0]; mk = R['choice'][g]['model'].replace('model_', '')
panels = [('1.6 mm HCP map (reference)', E['z16_ccw'], E['z16_cw']), (f'{g} mm standard', E[f'z_{g}_ccw'], E[f'z_{g}_cw']),
          (f'{g} mm + smoothing ({R["choice"][g]["smooth"][6:]} mm)', E[f'{R["choice"][g]["smooth"]}_{g}_ccw'], E[f'{R["choice"][g]["smooth"]}_{g}_cw']),
          (f'{g} mm model-based', E[f'render_{g}_{mk}_ccw'], E[f'render_{g}_{mk}_cw'])]
wk = {hh: {d.meta['Name']: np.asarray(d.data) for d in nib.load(f'{W_ROOT}/results/week2/subjects/{S}/{S}.{hh}.week2_wedge.func.gii').darrays} for hh in 'LR'}
coh = np.concatenate([wk['L']['coherence_1.6_all'], wk['R']['coherence_1.6_all']])[idx]
fn = np.cross(Vi[Fi[:, 1]] - Vi[Fi[:, 0]], Vi[Fi[:, 2]] - Vi[Fi[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
occ = Vi[:, 1] < np.percentile(Vi[:, 1], 35); Fm = Fi[occ[Fi].all(1) & (fn[:, 0] > 0.2)]
fig, ax = plt.subplots(1, 4, figsize=(12, 4.6))
for k, (title, zc, zw) in enumerate(panels):
    th = polar_angle(zc.astype(complex), zw.astype(complex), delay)
    full = np.full(nv, np.nan); sel = (idx < nvL) & (coh > 0.25); full[idx[sel]] = th[sel]
    fv = np.angle(np.nanmean(np.exp(1j * full[Fm]), axis=1)); ok = np.isfinite(fv)
    ax[k].tripcolor(Triangulation(Vi[:, 1], Vi[:, 2], Fm), facecolors=np.zeros(len(Fm)), cmap='Greys', vmin=-4, vmax=1)
    ax[k].tripcolor(Triangulation(Vi[:, 1], Vi[:, 2], Fm, mask=~ok), facecolors=np.where(ok, fv, 0), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax[k].set_aspect('equal'); ax[k].set_axis_off(); ax[k].set_title(title, fontsize=8.5)
fig.suptitle(f'Subject {S}, left hemisphere, medial occipital (posterior left): wedge polar angle, coherence > 0.25', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW3-4_example_maps.png', dpi=170, bbox_inches='tight'); plt.close()

# ---------------- report
f = lambda x, p=3: f'{x[0]:.{p}f} ± {x[1]:.{p}f}'
L = ['# MVE week 3 — model-based estimation from emulated 2.4 / 3.2 mm data', '',
     f"Tuning subjects (fixed in advance: first five MVE subjects by ID): {', '.join(T['subjects'])}. Test subjects: {len(subs)}.", '',
     '## Settings chosen on the tuning subjects', '', '| grid | criterion | model (range, s.d. scale) | smoothing | upsampling + smoothing |', '|---|---|---|---|---|']
for g in GR:
    c = R['choice'][g]
    L.append(f"| {g} mm | overall (all coherent) | {c['model']} | {c['smooth']} | {c.get('upsmooth', '')} |")
    if 'model_leak' in c:
        L.append(f"| {g} mm | where unmixing is needed (clean at 1.6, leak > 10%) | {c['model_leak']} | {c['smooth_leak']} | {c.get('upsmooth_leak', '')} |")
L += ['', '## Test subjects: cross-run residual against the other run\'s 1.6 mm HCP map (lower is better)', '',
      '| grid | vertices | n per subject | ' + ' | '.join(MLAB[m] for m in MLAB) + ' | model better than smoothing / than upsampling+smoothing (subjects) |',
      '|---|---|---|' + '---|' * len(MLAB) + '---|']
for g in GR:
    for s in SLAB:
        n = ms([R['scores'][S][g]['_n'][s] for S in subs])
        d = arr(g, 'model', s) - arr(g, 'smoothing', s); d2 = arr(g, 'model', s) - arr(g, 'upsample_smoothing', s)
        L.append(f"| {g} mm | {SLAB[s]} | {n[0]:.0f} | " + ' | '.join(f(ms(arr(g, m, s))) for m in MLAB) +
                 f" | {int(np.nansum(d < 0))}/{int(np.isfinite(d).sum())} / {int(np.nansum(d2 < 0))}/{int(np.isfinite(d2).sum())} |")
L += ['', 'Secondary (settings tuned where unmixing is needed, clean at 1.6 mm and leak > 10%):', '',
      '| grid | vertices | smoothing (high-leak tuned) | model (high-leak tuned) | model better (subjects) |', '|---|---|---|---|---|']
for g in GR:
    for s_ in SLAB:
        if not all('model_leak' in R['scores'][S][g] and R['scores'][S][g]['model_leak'] for S in subs):
            continue
        d = arr(g, 'model_leak', s_) - arr(g, 'smoothing_leak', s_)
        L.append(f"| {g} mm | {SLAB[s_]} | {f(ms(arr(g, 'smoothing_leak', s_)))} | {f(ms(arr(g, 'model_leak', s_)))} | {int(np.nansum(d < 0))}/{int(np.isfinite(d).sum())} |")
if all(R['scores'][S][g].get('model_psf1.15') for S in subs for g in GR):
    L += ['', '## Robustness to a mis-specified PSF (the inversion uses coarse-grid kernels 15% too narrow or too wide)', '',
          '| grid | vertices | model, PSF -15% | model, exact PSF | model, PSF +15% | best competitor (smoothing / upsampling+smoothing) |',
          '|---|---|---|---|---|---|']
    for g in GR:
        for s_ in SLAB:
            best = np.fmin(arr(g, 'smoothing', s_), arr(g, 'upsample_smoothing', s_))
            L.append(f"| {g} mm | {SLAB[s_]} | {f(ms(arr(g, 'model_psf0.85', s_)))} | {f(ms(arr(g, 'model', s_)))} | "
                     f"{f(ms(arr(g, 'model_psf1.15', s_)))} | {f(ms(best))} |")
L += ['', '## Opposite-bank contrast kept (dissimilarity of polar angle across the sulcus, relative to the 1.6 mm map)', '',
      '| grid | pairs | ' + ' | '.join(MLAB[m] for m in MLAB) + ' |', '|---|---|' + '---|' * len(MLAB)]
for g in GR:
    for suf, lab in (('', 'all coherent pairs'), ('_calc', 'pericalcarine pairs')):
        L.append(f'| {g} mm | {lab} | ' + ' | '.join(f(ms([R["pairs"][S][f"{g}_{m}{suf}"] / R["pairs"][S][f"ref_1.6{suf}"] for S in subs]), 2) for m in MLAB) + ' |')
L += ['', f"Pairs per subject: {ms([R['pairs'][S]['n_pairs'] for S in subs])[0]:.0f} (pericalcarine {ms([R['pairs'][S]['n_pairs_calc'] for S in subs])[0]:.0f}).", '',
      '## Averaging kernels at sample vertices (half with > 10% standard-mapping leak)', '',
      '| grid | vertices | net leak, standard | net leak, model | gross far weight, standard | gross far weight, model | spread along the sheet (mm), standard | spread, model |',
      '|---|---|---|---|---|---|---|---|']
for g in GR:
    rows = [r for S in subs for r in R['kernels'].get(S, {}).get(g, [])]
    if rows:
        q = lambda k, sc=100: ms([r[k] * sc for r in rows])
        L.append(f"| {g} mm | {len(rows)} | {f(q('std_leak_fraction'), 1)}% | {f(q('model_leak_fraction'), 1)}% | {f(q('std_leak_abs_fraction'), 1)}% | "
                 f"{f(q('model_leak_abs_fraction'), 1)}% | {f(q('std_spread_mm', 1), 2)} | {f(q('model_spread_mm', 1), 2)} |")
L += ['', '## Known-truth simulation (forward model exact; residual against the noise-free 1.6 mm rendering of the truth)', '',
      '| grid | vertices | ' + ' | '.join(MLAB[m] for m in MLAB) + ' |', '|---|---|' + '---|' * len(MLAB)]
for g in GR:
    for s in SLAB:
        L.append(f'| {g} mm | {SLAB[s]} | ' + ' | '.join(f(ms([R["simulation"][S][g][m][s] for S in subs if g in R["simulation"].get(S, {})]), 4) for m in MLAB) + ' |')
L += ['', 'Figures: figW3-1_tuning.png, figW3-2_test_scores.png, figW3-3_banks_kernels_sim.png, figW3-4_example_maps.png.', '',
      'Method: posterior mean of the column-mean field on all cortex under a Matern (nu = 1) surface prior (SPDE/FEM), per-voxel noise from',
      'neighbouring frequencies, K from week 2 (effective PSF, both hemispheres); estimates are rendered through the 1.6 mm model and HCP',
      'mapping (W_1.6 K_1.6 f) and compared with the other wedge run\'s standard 1.6 mm map after the best complex scaling (conjugated),',
      'so that estimate and reference noise are independent. Smoothing = implicit heat-equation smoothing on the surface.']
open(f'{OUT}/week3_report.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
