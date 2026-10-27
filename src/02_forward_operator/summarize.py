"""MVE week 2: aggregate per-subject results into tables, figures and a report.
Usage (xcp_d container): python3 summarize.py WEEK2_SUBJECTS_DIR WEEK1_RESULTS_DIR OUTDIR
"""
import os, sys, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

SUBJ, W1, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
COMBOS = ['1.6_gv', '1.6_all', '2.4_all', '3.2_all']
LAB = {'1.6_gv': '1.6 mm (HCP standard, goodvoxels)', '1.6_all': '1.6 mm (no voxel mask)', '2.4_all': '2.4 mm emulated',
       '3.2_all': '3.2 mm emulated'}
SHORT = {'1.6_gv': '1.6 gv', '1.6_all': '1.6', '2.4_all': '2.4', '3.2_all': '3.2'}
COL = {'1.6_gv': '#4C72B0', '1.6_all': '#8FAADC', '2.4_all': '#DD8452', '3.2_all': '#C44E52'}
BINC = {'0-2.25': 2.0, '2.25-2.75': 2.5, '2.75-3.25': 3.0, '3.25-3.75': 3.5, '3.75-4.5': 4.0, '4.5-6': 5.0, '6-inf': 7.0}
STRATA = {'coherent': 'all coherent vertices', 'coherent_leakfrac_lt2': 'model leak fraction < 2%',
          'coherent_leakfrac_gt5': 'model leak fraction > 5%', 'coherent_leakfrac_gt10': 'model leak fraction > 10%',
          'coherent_calc': 'coherent pericalcarine'}
PRED = ['naive', 'model_near', 'model_full', 'model_far_shuffled']
PLAB = {'naive': '1.6 mm map as is', 'model_near': 'operator, near part', 'model_full': 'operator, near + far',
        'model_far_shuffled': 'near + shuffled far'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def load_maps(p):
    return {da.meta['Name']: np.asarray(da.data, np.float64) for da in nib.load(p).darrays}


def ms(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(np.std(x))


subs = sorted(d for d in os.listdir(SUBJ) if os.path.exists(f'{SUBJ}/{d}/{d}.week2_summary.json'))
J = {S: json.load(open(f'{SUBJ}/{S}/{S}.week2_summary.json')) for S in subs}
CH = {S: json.load(open(f'{SUBJ}/{S}/{S}.week2_checks.json')) for S in subs if os.path.exists(f'{SUBJ}/{S}/{S}.week2_checks.json')}
print('subjects:', len(subs), '| with operator checks:', list(CH))

# ---------------- per-hemisphere table
OPK = ('frac_cortex_no_weights', 'dilution_mean', 'own_mean', 'leak_mean', 'leak_abs_mean', 'leak_other_hemi_mean', 'leak_frac_mean',
       'frac_leak_frac_gt2', 'frac_leak_frac_gt5', 'frac_leak_frac_gt10', 'calc_dilution_mean', 'calc_leak_mean', 'calc_leak_abs_mean',
       'calc_leak_frac_mean', 'calc_frac_leak_frac_gt5', 'r_week1_leak7T')
WK = ('median_coh_calc', 'median_coh_cortex', 'frac_cortex_coherent', 'agreement_with_1p6', 'median_abs_dtheta_vs_1p6_deg',
      'median_abs_dtheta_vs_1p6_calc_deg', 'tsnr_median_calc')
rows = []
for S in subs:
    j = J[S]
    for h in 'LR':
        r = {'subject': S, 'hemi': h, 'n_coherent': j['wedge'][h]['n_coherent'], 'n_coherent_calc': j['wedge'][h]['n_coherent_calc']}
        for c in COMBOS:
            g, tag = c.split('_')
            o = j['operator'][g][f'{tag}_{h}']
            for k in OPK:
                r[f'{c}:{k}'] = o[k]
            for b, v in o['by_week1_dopp'].items():
                r[f'{c}:leak_bin_{b}'] = v['leak']; r[f'{c}:leakabs_bin_{b}'] = v['leak_abs']; r[f'{c}:own_bin_{b}'] = v['own']
                r[f'area_bin_{b}'] = v['area_frac']
            for k in WK:
                r[f'{c}:wedge_{k}'] = j['wedge'][h][c][k]
        for g in ('2.4', '3.2'):
            for s, v in j['prediction'][f'{g}_{h}'].items():
                r[f'pred_{g}:{s}:n'] = v['n']
                for p in PRED:
                    r[f'pred_{g}:{s}:{p}'] = v[p]
        for k, v in j['geometry_check'][h].items():
            r[f'geom:{k}'] = v
        rows.append(r)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('subject', 'hemi'), k))
with open(f'{OUT}/week2_per_hemisphere.csv', 'w') as fh:
    fh.write(','.join(keys) + '\n')
    for r in rows:
        fh.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
G = {k: ms([r[k] for r in rows if k in r and r[k] == r[k]]) for k in keys if k not in ('subject', 'hemi')}
G['_n_hemispheres'] = len(rows)
G['_psf_fwhm_mm_xyz'] = {g: np.mean([J[S]['operator'][g]['psf_fwhm_mm_xyz'] for S in subs], 0).round(2).tolist() for g in ('1.6', '2.4', '3.2')}
G['_nnz_K_millions'] = {g: ms([J[S]['operator'][g]['nnz_K'] / 1e6 for S in subs]) for g in ('1.6', '2.4', '3.2')}
G['_t_total_min'] = ms([J[S]['t_total_s'] / 60 for S in subs])
G['_mni_over_acpc_scales'] = np.mean([J[S]['mni_over_acpc_scales'] for S in subs], 0).round(3).tolist()
G['_voxels_per_vertex'] = {c: ms([J[S]['voxels_per_vertex'][c] for S in subs]) for c in COMBOS}
G['_delay_rad_mod_pi'] = ms([J[S]['wedge']['delay_rad_mod_pi'] for S in subs])
# paired test of the far part: hemispheres where near+far beats near, and where near+shuffled beats near
for g in ('2.4', '3.2'):
    for s in STRATA:
        kk = [f'pred_{g}:{s}:{p}' for p in PRED]
        v = np.array([[r.get(k, np.nan) for k in kk] for r in rows], float); v = v[np.isfinite(v).all(1)]
        if len(v):
            G[f'_pred_{g}:{s}:full_beats_near'] = int((v[:, 2] < v[:, 1]).sum())
            G[f'_pred_{g}:{s}:shuffled_beats_near'] = int((v[:, 3] < v[:, 1]).sum())
            G[f'_pred_{g}:{s}:n_hemi'] = int(len(v))
if CH:
    G['_checks'] = {S: CH[S] for S in CH}
json.dump(G, open(f'{OUT}/week2_group_mean_sd.json', 'w'), indent=1)

# ---------------- week-1 lookup curves (large-scale pattern, q index 0)
look = json.load(open(f'{W1}/lookup/lookup_banks.json'))['results']
w1c = {a: sorted([(r['d'], r['leak_naive'][0], r['own_naive'][0]) for r in look if r['acq'] == a and r['cnr'] == 1.0]) for a in ('3T', '7T')}

# ---------------- figure 1: leakage vs opposite-bank distance, distribution over cortex, calcarine
pool = {c: [] for c in COMBOS}
for S in subs:
    for h in 'LR':
        m = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_leakage.func.gii')
        roi = m['dilution_1.6_all'] != 0
        for c in COMBOS:
            g, tag = c.split('_')
            pool[c].append(m[f'leak_fraction_{g}_{tag}'][roi])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
xs = [BINC[b] for b in BINC]
for c in COMBOS:
    y = np.array([[r[f'{c}:leak_bin_{b}'] for b in BINC] for r in rows]) * 100
    ax[0].errorbar(xs, y.mean(0), yerr=y.std(0), color=COL[c], marker='o', ms=3, capsize=2, label=LAB[c])
    if c in ('2.4_all', '3.2_all'):
        ya = np.array([[r[f'{c}:leakabs_bin_{b}'] for b in BINC] for r in rows]) * 100
        ax[0].plot(xs, ya.mean(0), color=COL[c], ls=':', lw=1)
for a, ls in (('7T', '--'), ('3T', '-.')):
    d, lk, _ = zip(*w1c[a]); ax[0].plot(d, 100 * np.array(lk), 'k' + ls, lw=1, label=f'week-1 planar-bank proxy, {a}')
ax[0].set_xlabel('distance to nearest opposite bank (week 1, mm)'); ax[0].set_ylabel('readout from far cortex (% of a unit field)')
ax[0].set_title('Leakage of standard ribbon mapping\n(dotted: magnitude, before sidelobe cancellation)', fontsize=9)
ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_xlim(1.8, 7.2)
for c in COMBOS:
    x = np.sort(np.concatenate(pool[c])) * 100
    ax[1].plot(x, 1 - np.arange(len(x)) / len(x), color=COL[c], label=LAB[c])
ax[1].set_xscale('log'); ax[1].set_xlim(0.1, 60); ax[1].set_xlabel('leak fraction at a vertex (% of its readout)')
ax[1].set_ylabel('fraction of cortical vertices exceeding'); ax[1].set_title(f'Distribution over cortex ({len(subs)} subjects)')
xb = np.arange(len(COMBOS)); wd = 0.38
for k, (key, lab, off) in enumerate((('leak_frac_mean', 'whole cortex', -wd / 2), ('calc_leak_frac_mean', 'pericalcarine', wd / 2))):
    v = np.array([[r[f'{c}:{key}'] for c in COMBOS] for r in rows]) * 100
    ax[2].bar(xb + off, v.mean(0), wd, yerr=v.std(0), capsize=2, color=['#999999', '#555555'][k], label=lab)
ax[2].set_xticks(xb); ax[2].set_xticklabels([SHORT[c] for c in COMBOS]); ax[2].set_ylabel('mean leak fraction (%)')
ax[2].set_title('Whole cortex vs pericalcarine'); ax[2].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/figW2-1_leakage.png', dpi=170); plt.close()

# ---------------- figures 2 and 3 on the example subject's very inflated surface (generated in week 1)
S = subs[0]; h = 'L'
vinfl = f'{W1}/maps/_summary/{S}.{h}.very_inflated.surf.gii'


def draw(axx, Vi, Fi, vals, cmap, lo, hi, view, cyclic=False):
    fv = (np.angle(np.nanmean(np.exp(1j * vals[Fi]), axis=1)) if cyclic else np.nanmean(vals[Fi], axis=1))
    ok = np.isfinite(np.nanmean(vals[Fi], axis=1))
    col = plt.get_cmap(cmap)((np.clip(fv, lo, hi) - lo) / (hi - lo)); col[~ok] = (0.85, 0.85, 0.85, 1)
    tri = axx.plot_trisurf(Vi[:, 0], Vi[:, 1], Vi[:, 2], triangles=Fi, linewidth=0, antialiased=False, shade=False)
    tri.set_facecolor(col); tri.set_edgecolor('none')
    axx.view_init(elev=view[0], azim=view[1]); axx.set_axis_off(); axx.set_box_aspect(np.ptp(Vi[np.unique(Fi)], axis=0))


if os.path.exists(vinfl):
    gi = nib.load(vinfl); Vi = gi.agg_data('NIFTI_INTENT_POINTSET'); Fi = gi.agg_data('NIFTI_INTENT_TRIANGLE')
    m = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_leakage.func.gii')
    roi = m['dilution_1.6_all'] != 0
    panels = [('dilution_1.6_gv', 'partial-volume dilution, 1.6 mm', 'viridis', 0.3, 1.0),
              ('leak_fraction_1.6_gv', 'leak fraction, 1.6 mm', 'Reds', 0.0, 0.25),
              ('leak_fraction_2.4_all', 'leak fraction, 2.4 mm emulated', 'Reds', 0.0, 0.25),
              ('leak_fraction_3.2_all', 'leak fraction, 3.2 mm emulated', 'Reds', 0.0, 0.25)]
    fig = plt.figure(figsize=(13, 6))
    for i, (k, title, cmap, lo, hi) in enumerate(panels):
        for jv, view in enumerate([(0, 180), (0, 0)]):
            axx = fig.add_subplot(2, 4, i + 1 + 4 * jv, projection='3d')
            draw(axx, Vi, Fi, np.where(roi, m[k], np.nan), cmap, lo, hi, view)
            if jv == 0:
                axx.set_title(title, fontsize=9)
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)); sm.set_array([])
                fig.colorbar(sm, ax=axx, orientation='horizontal', fraction=0.05, pad=0.02)
    fig.suptitle(f'Subject {S}, left hemisphere (very inflated; top lateral, bottom medial)', fontsize=10)
    plt.tight_layout(); plt.savefig(f'{OUT}/figW2-2_example_leakage_maps.png', dpi=150); plt.close()

    mw = load_maps(f'{SUBJ}/{S}/{S}.{h}.week2_wedge.func.gii')
    fig = plt.figure(figsize=(13, 7.4))
    # medial occipital surface, flattened by projection onto the y-z plane (faces of the posterior cap facing the midline)
    fn = np.cross(Vi[Fi[:, 1]] - Vi[Fi[:, 0]], Vi[Fi[:, 2]] - Vi[Fi[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    occ = Vi[:, 1] < np.percentile(Vi[:, 1], 35)
    Fm = Fi[occ[Fi].all(1) & ((1 if h == 'L' else -1) * fn[:, 0] > 0.2)]
    for i, c in enumerate(['1.6_gv', '2.4_all', '3.2_all']):
        ax = fig.add_subplot(2, 3, i + 1)
        vals = np.where(roi & (mw[f'coherence_{c}'] > 0.25), mw[f'theta_{c}'], np.nan)
        fv = np.angle(np.nanmean(np.exp(1j * vals[Fm]), axis=1)); ok = np.isfinite(fv)
        ax.tripcolor(Triangulation(Vi[:, 1], Vi[:, 2], Fm), facecolors=np.zeros(len(Fm)), cmap='Greys', vmin=-4, vmax=1)
        ax.tripcolor(Triangulation(Vi[:, 1], Vi[:, 2], Fm, mask=~ok), facecolors=np.where(ok, fv, 0), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_title(f'wedge polar angle, {LAB[c]}\n(medial occipital, posterior left; coherence > 0.25)', fontsize=8.5)
    ax = fig.add_subplot(2, 3, 4)
    for key, lab, cc in (('wedge_median_coh_calc', 'pericalcarine', '#555555'), ('wedge_median_coh_cortex', 'whole cortex', '#aaaaaa')):
        v = np.array([[r[f'{c}:{key}'] for c in COMBOS] for r in rows])
        ax.errorbar(range(len(COMBOS)), v.mean(0), yerr=v.std(0), marker='o', capsize=3, color=cc, label=lab)
    ax.set_xticks(range(len(COMBOS))); ax.set_xticklabels([SHORT[c] for c in COMBOS]); ax.set_ylabel('median wedge coherence')
    ax.legend(frameon=False); ax.set_title('Response coherence (standard mapping)')
    ax = fig.add_subplot(2, 3, (5, 6))
    groups = [(g, s) for g in ('2.4', '3.2') for s in ('coherent', 'coherent_leakfrac_lt2', 'coherent_leakfrac_gt5', 'coherent_leakfrac_gt10')]
    wbar = 0.2
    for pi_, p in enumerate(PRED):
        v = np.array([[r.get(f'pred_{g}:{s}:{p}', np.nan) for (g, s) in groups] for r in rows], float)
        ax.bar(np.arange(len(groups)) + (pi_ - 1.5) * wbar, np.nanmean(v, 0), wbar, yerr=np.nanstd(v, 0), capsize=1.5,
               color=['#bbbbbb', '#8FAADC', '#4C72B0', '#DDAA88'][pi_], label=PLAB[p])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g} mm\n{STRATA[s].replace('model leak fraction', 'leak')}" for g, s in groups], fontsize=7.5)
    ax.set_ylabel('residual energy fraction'); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title('Predicting the emulated maps from the 1.6 mm map through the operator (mean ± SD over hemispheres)')
    plt.tight_layout(); plt.savefig(f'{OUT}/figW2-3_wedge_prediction.png', dpi=150); plt.close()


# ---------------- report
def f(k, p=2, s=1.0):
    m, sd = G[k]
    return f'{m * s:.{p}f} ± {sd * s:.{p}f}'


L = ['# MVE week 2 — forward operator on real geometry, HCP ribbon weights, emulated 2.4 / 3.2 mm acquisitions', '',
     f'Subjects: {len(subs)} (MVE calibration families), both hemispheres; HCP 7T retinotopy, MNINonLinear space.', '',
     '## Operator', '',
     '| grid | effective PSF FWHM x / y / z (mm) | nonzeros in K (millions) | median voxels per vertex (HCP weights) |',
     '|---|---|---|---|']
for g, c in (('1.6', '1.6_all'), ('2.4', '2.4_all'), ('3.2', '3.2_all')):
    fw = G['_psf_fwhm_mm_xyz'][g]
    L.append(f"| {g} mm | {fw[0]:.2f} / {fw[1]:.2f} / {fw[2]:.2f} | {G['_nnz_K_millions'][g][0]:.0f} | {G['_voxels_per_vertex'][c][0]:.0f} |")
L += ['', f"MNI/ACPC scale x/y/z {G['_mni_over_acpc_scales']}; HCP goodvoxels (RETCCW) leave {f('1.6_gv:frac_cortex_no_weights', 2, 100)}% of cortex "
      f"without usable voxels (excluded from the averages); run time per subject {G['_t_total_min'][0]:.1f} ± {G['_t_total_min'][1]:.1f} min.", '']
if CH:
    L += ['### Checks (test subjects)', '', '| subject | grid | brute-force max rel. error | column mass p1 / p50 / p99 |', '|---|---|---|---|']
    for S_, c in CH.items():
        for g, v in c['check1'].items():
            L.append(f"| {S_} | {g} | {v['max_rel_err']:.1e} | {' / '.join(f'{x:.3f}' for x in v['mass_p01_p50_p99'])} |")
    L += ['', 'Emulation consistency, S(K_1.6 f) vs K_g f (relative L2 error on the operator support / after standard mapping):', '',
          '| subject | field | 2.4 mm | 3.2 mm |', '|---|---|---|---|']
    for S_, c in CH.items():
        for name, v in c['check3'].items():
            if name.startswith('single'):
                continue
            L.append(f"| {S_} | {name} | {v['2.4']['rel_err_operator_support']:.3f} / {v['2.4']['rel_err_mapped_vertices']:.3f} | "
                     f"{v['3.2']['rel_err_operator_support']:.3f} / {v['3.2']['rel_err_mapped_vertices']:.3f} |")
        sg = [v for n, v in c['check3'].items() if n.startswith('single')]
        L.append(f"| {S_} | single columns (median of {len(sg)}) | {np.median([v['2.4']['rel_err_operator_support'] for v in sg]):.3f} | "
                 f"{np.median([v['3.2']['rel_err_operator_support'] for v in sg]):.3f} |")
    L += ['', 'Depth-gradient component b1 (native grid), median |net| / gross column mass: ' +
          ', '.join(f"{S_} {c['b1_native']['median_net_over_gross']:.3f}" for S_, c in CH.items()), '']
L += ['## Geometry check: box-overlap proxy (week 1) vs Workbench ribbon weights (1.6 mm, no mask)', '',
      f"- median per-vertex weight overlap {f('geom:median_weight_overlap', 3)}; median per-vertex weight correlation {f('geom:median_r', 3)}",
      f"- voxel occupancy correlation {f('geom:voxel_occupancy_r', 3)}; voxels per vertex: box {f('geom:voxels_per_vertex_box', 1)}, "
      f"Workbench {f('geom:voxels_per_vertex_wb', 1)}", '',
      '## What standard ribbon mapping does on real geometry (A = W K, unit field)', '',
      '| | dilution (own + leak) | leak (% of unit field) | of which other hemisphere | leak magnitude | leak fraction (%) | cortex with leak fraction > 5% / > 10% | pericalcarine leak fraction (%) |',
      '|---|---|---|---|---|---|---|---|']
for c in COMBOS:
    L.append(f"| {LAB[c]} | {f(c + ':dilution_mean', 3)} | {f(c + ':leak_mean', 2, 100)} | {f(c + ':leak_other_hemi_mean', 2, 100)} | "
             f"{f(c + ':leak_abs_mean', 2, 100)} | {f(c + ':leak_frac_mean', 2, 100)} | "
             f"{f(c + ':frac_leak_frac_gt5', 1, 100)}% / {f(c + ':frac_leak_frac_gt10', 1, 100)}% | {f(c + ':calc_leak_frac_mean', 2, 100)} |")
L += ['', 'Leak (% of a unit far field) by week-1 opposite-bank distance, and the week-1 planar-bank proxy:', '',
      '| d_opp bin (mm) | area share | ' + ' | '.join(LAB[c] for c in COMBOS) + ' | week-1 proxy 7T | week-1 proxy 3T |',
      '|---|---|' + '---|' * len(COMBOS) + '---|---|']
p7 = dict((d, lk) for d, lk, _ in w1c['7T']); p3 = dict((d, lk) for d, lk, _ in w1c['3T'])
for b, x in BINC.items():
    prox = f" | {100 * p7[x]:.2f} | {100 * p3[x]:.2f} |" if x in p7 else " | – | – |"
    L.append(f"| {b} | {G[f'area_bin_{b}'][0] * 100:.1f}% | " + ' | '.join(f(f'{c}:leak_bin_{b}', 2, 100) for c in COMBOS) + prox)
L += ['', 'Leak magnitude (sum of |A| over far cortex, % of a unit field; what a patterned far field can carry before positive and negative',
      'sidelobe contributions cancel) by the same bins:', '',
      '| d_opp bin (mm) | ' + ' | '.join(LAB[c] for c in COMBOS) + ' |', '|---|' + '---|' * len(COMBOS)]
for b in BINC:
    L.append(f"| {b} | " + ' | '.join(f(f'{c}:leakabs_bin_{b}', 2, 100) for c in COMBOS) + ' |')
L += ['', f"Per-vertex correlation with the week-1 7T proxy map (1.6 mm, no mask): {f('1.6_all:r_week1_leak7T', 3)}; "
      f"with goodvoxels: {f('1.6_gv:r_week1_leak7T', 3)}.", '',
      '## Wedge data (RETCCW + RETCW, standard mapping)', '',
      '| | median coherence, pericalcarine | median coherence, cortex | cortex with coherence > 0.3 | tSNR, pericalcarine | agreement with 1.6 mm (resultant) | median polar-angle change vs 1.6 mm, coherent / pericalcarine (deg) |',
      '|---|---|---|---|---|---|---|']
for c in COMBOS:
    L.append(f"| {LAB[c]} | {f(c + ':wedge_median_coh_calc', 3)} | {f(c + ':wedge_median_coh_cortex', 3)} | {f(c + ':wedge_frac_cortex_coherent', 1, 100)}% | "
             f"{f(c + ':wedge_tsnr_median_calc', 1)} | {f(c + ':wedge_agreement_with_1p6', 3)} | "
             f"{f(c + ':wedge_median_abs_dtheta_vs_1p6_deg', 1)} / {f(c + ':wedge_median_abs_dtheta_vs_1p6_calc_deg', 1)} |")
L += ['', f"Coherent vertices (> 0.3 at 1.6 mm): {f('n_coherent', 0)} per hemisphere, {f('n_coherent_calc', 0)} of them pericalcarine; "
      f"haemodynamic delay (mod pi) {G['_delay_rad_mod_pi'][0]:.2f} ± {G['_delay_rad_mod_pi'][1]:.2f} rad.", '',
      '## Cross-resolution prediction: does the operator predict how the maps change?', '',
      'Residual energy fraction of the emulated map after the best complex scaling of each predictor (both wedge runs pooled);',
      'mean ± SD over hemispheres, and the number of hemispheres in which adding the far part (or a shuffled far part) beats the near part alone.', '',
      '| grid | vertices | n per hemisphere | ' + ' | '.join(PLAB[p] for p in PRED) + ' | near + far better | near + shuffled better |',
      '|---|---|---|' + '---|' * len(PRED) + '---|---|']
for g in ('2.4', '3.2'):
    for s, lab in STRATA.items():
        if f'pred_{g}:{s}:naive' not in G:
            continue
        nh = G.get(f'_pred_{g}:{s}:n_hemi', 0)
        L.append(f"| {g} mm | {lab} | {f(f'pred_{g}:{s}:n', 0)} | " + ' | '.join(f(f'pred_{g}:{s}:{p}', 4) for p in PRED) +
                 f" | {G.get(f'_pred_{g}:{s}:full_beats_near', 0)}/{nh} | {G.get(f'_pred_{g}:{s}:shuffled_beats_near', 0)}/{nh} |")
L += ['', 'Figures: figW2-1_leakage.png, figW2-2_example_leakage_maps.png, figW2-3_wedge_prediction.png. '
      'Tables: week2_per_hemisphere.csv, week2_group_mean_sd.json.', '',
      'Model: effective PSF = native 7T PSF (Tukey 0.3 in-plane, T2* along PE (y), slice profile, 0.4 mm resampling blur; MNI/ACPC-scaled),',
      'truncated at 4/4/3 voxels; emulated grids = central k-space block x Tukey 0.5 window (zero at the coarse Nyquist), operators at',
      '5 voxels support. Signal only from grey matter (column mean, b0). Far cortex = other hemisphere or > 15 mm geodesic on the sphere.',
      'Emulated acquisitions keep the 7T noise and physiological structure (k-space cropping of the acquired data), so their SNR gain',
      'is at most sqrt(voxel-volume ratio), less than a native acquisition at that voxel size would give.']
open(f'{OUT}/week2_report.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
