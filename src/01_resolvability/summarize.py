"""MVE week 1, part 3: aggregate per-subject maps into tables, figures and a short report.

Usage (inside the xcp_d container): python3 summarize.py MAPSDIR LOOKUPDIR THEORY_RESULTS_DIR
"""
import os, sys, json, glob, subprocess
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAPS, LOOK, THEO = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(MAPS, '_summary'); os.makedirs(OUT, exist_ok=True)
ACQS = ['3T', '7T']; CFGS = ['white_cnr1', 'white_cnr10', 'bold_cnr10']
CFG_LAB = {'white_cnr1': 'acquisition-limited, single frame (CNR 1)', 'white_cnr10': 'acquisition-limited, map level (CNR 10)',
           'bold_cnr10': 'with BOLD blur, map level (CNR 10)'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'ideal': '#555555', 'white_cnr1': '#4C72B0', 'white_cnr10': '#55A868', 'bold_cnr10': '#DD8452'}


def load_maps(p):
    g = nib.load(p)
    return {da.meta['Name']: np.asarray(da.data) for da in g.darrays}


def areas(surf):
    g = nib.load(surf)
    V = g.agg_data('NIFTI_INTENT_POINTSET').astype(float); F = g.agg_data('NIFTI_INTENT_TRIANGLE').astype(np.int64)
    fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    va = np.zeros(len(V))
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3)
    return va


subs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(MAPS, '[0-9]' * 6)))
rows, pooled = [], {}
for S in subs:
    js = json.load(open(os.path.join(MAPS, S, f'{S}.week1_summary.json')))
    for h in 'LR':
        s = js[h]
        row = dict(subject=S, hemi=h, area_cm2=s['area_cm2_acpc'], thickness_median=s['thickness_median'],
                   pw_thick_acpc=s['pw_over_thick_acpc'], pw_thick_mni=s['pw_over_thick_mni'],
                   r_acpc=s['r_pw_thick_acpc'], r_mni=s['r_pw_thick_mni'], dopp_median=s['dopp_median'])
        for thr in (3, 4, 5, 6):
            row[f'frac_dopp_lt{thr}'] = s[f'frac_dopp_lt{thr}']
        for acq in ACQS:
            row[f'lam_ideal_{acq}_med'] = s[f'lam_ideal_{acq}_q'][2]
            for cfg in CFGS:
                row[f'lam_{acq}_{cfg}_med'] = s[f'lam_{acq}_{cfg}_q'][2]
                row[f'dfs_{acq}_{cfg}'] = s[f'dfs_hemi_{acq}_{cfg}']
            row[f'leak_naive_{acq}_mean'] = s[f'leak_naive_{acq}_mean']
            for c in (1, 10):
                for tag in ('tau8', 'tau5', 'tau4'):
                    row[f'unrec_{tag}_{acq}_cnr{c}'] = s[f'frac_unrec_{tag}_{acq}_cnr{c}']
        rows.append(row)
        m = load_maps(os.path.join(MAPS, S, f'{S}.{h}.week1_maps.func.gii'))
        a = areas(os.path.join(MAPS, S, f'{S}.{h}.midthickness.acpc.surf.gii'))
        cortex = m['thickness'] > 0
        for k, v in m.items():
            pooled.setdefault(k, []).append(v[cortex])
        pooled.setdefault('_area', []).append(a[cortex])
pooled = {k: np.concatenate(v) for k, v in pooled.items()}
wa = pooled['_area'] / pooled['_area'].sum()

keys = list(rows[0].keys())
with open(os.path.join(OUT, 'week1_per_hemisphere.csv'), 'w') as f:
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(str(r[k]) if isinstance(r[k], str) else f'{r[k]:.4g}' for k in keys) + '\n')
num = [k for k in keys if k not in ('subject', 'hemi')]
G = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in num}
json.dump(G, open(os.path.join(OUT, 'week1_group_mean_sd.json'), 'w'), indent=1)

# fsaverage5 reference (idealised formula, 2 mm; scale to 1.6 mm)
g5 = np.load(os.path.join(THEO, 'gain_iso2.npy')); a5 = np.load(os.path.join(THEO, 'fs5_areas.npy')); w5 = a5 / a5.sum()
fs5 = {'3T': 4.0 / g5, '7T': 3.2 / g5}


def wquant(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, q * cw[-1])])


# ---------------- figure W1: resolvable wavelength distributions
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for j, acq in enumerate(ACQS):
    bins = np.linspace(1.5, 8.0, 66)
    ax[j].hist(pooled[f'lam_ideal_{acq}'], bins=bins, weights=wa * 100, histtype='step', lw=1.8, color=COL['ideal'], label='idealised formula, 20 subjects')
    ax[j].hist(fs5[acq], bins=bins, weights=w5 * 100, histtype='step', lw=1.2, ls='--', color='k', label='idealised formula, fsaverage5')
    for cfg in CFGS:
        ax[j].hist(pooled[f'lam_{acq}_{cfg}'], bins=bins, weights=wa * 100, histtype='step', lw=1.6, color=COL[cfg], label=CFG_LAB[cfg])
    h = 2.0 if acq == '3T' else 1.6
    ax[j].axvline(2 * h, color='k', lw=0.8, ls=':'); ax[j].text(2 * h, ax[j].get_ylim()[1] * 0.02, ' 2h', fontsize=7)
    ax[j].set_title(f'HCP {acq} ({h:g} mm voxels)'); ax[j].set_xlabel('resolvable isotropic tangential wavelength (mm)')
ax[0].set_ylabel('% of cortical area'); ax[1].legend(frameon=False, fontsize=7, loc='upper right')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'figW1_resolvable_wavelength.png'), dpi=170); plt.close()

# ---------------- figure W2: bank proximity, naive leakage, unrecoverable fraction
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
d = np.minimum(pooled['d_opp'], 9.0)
ax[0].hist(d, bins=np.linspace(0, 9, 46), weights=wa * 100, color='#8172B2')
ax[0].set_xlabel('distance to nearest opposite bank (mm; ≥9 pooled)'); ax[0].set_ylabel('% of cortical area')
ax[0].set_title('Sulcal-bank proximity (midthickness to midthickness)')
for acq, c in zip(ACQS, ['#C44E52', '#4C72B0']):
    ax[1].hist(pooled[f'leak_naive_{acq}'] * 100, bins=np.linspace(0, 20, 41), weights=wa * 100, histtype='step', lw=1.8, color=c, label=f'{acq}')
ax[1].set_yscale('log'); ax[1].set_xlabel('predicted cross-bank leakage of ribbon averaging (%)'); ax[1].set_ylabel('% of cortical area (log)')
ax[1].legend(frameon=False); ax[1].set_title('Standard-pipeline leakage (large-scale patterns)')
labs, v5, v4 = [], [], []
for acq in ACQS:
    for c in (1, 10):
        labs.append(f'{acq}\nCNR {c}')
        v5.append(np.mean([r[f'unrec_tau5_{acq}_cnr{c}'] * 100 for r in rows]))
        v4.append(np.mean([r[f'unrec_tau4_{acq}_cnr{c}'] * 100 for r in rows]))
xx = np.arange(4)
ax[2].bar(xx - 0.2, v5, width=0.4, color='#DD8452', label='5.2 mm patterns')
ax[2].bar(xx + 0.2, v4, width=0.4, color='#8C564B', label='3.9 mm patterns')
ax[2].set_xticks(xx); ax[2].set_xticklabels(labs); ax[2].set_ylabel('% of cortical area'); ax[2].set_ylim(0, 105)
ax[2].legend(frameon=False, fontsize=7)
ax[2].set_title('Bank-specific patterns NOT recoverable\n(optimal estimator, transfer < 0.5)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'figW2_banks.png'), dpi=170); plt.close()

# ---------------- figure W3: example subject maps on an inflated surface
S = subs[0]; h = 'L'
mid = os.path.join(MAPS, S, f'{S}.{h}.midthickness.acpc.surf.gii')
infl = os.path.join(OUT, f'{S}.{h}.inflated.surf.gii'); vinfl = os.path.join(OUT, f'{S}.{h}.very_inflated.surf.gii')
if not os.path.exists(infl):
    subprocess.run(['wb_command', '-surface-generate-inflated', mid, infl, vinfl], check=True)
gi = nib.load(vinfl); Vi = gi.agg_data('NIFTI_INTENT_POINTSET'); Fi = gi.agg_data('NIFTI_INTENT_TRIANGLE')
m = load_maps(os.path.join(MAPS, S, f'{S}.{h}.week1_maps.func.gii'))
panels = [('lam_3T_bold_cnr10', 'resolvable wavelength, 3T (mm)', 'viridis_r', 2.9, 4.1),
          ('lam_7T_bold_cnr10', 'resolvable wavelength, 7T (mm)', 'viridis_r', 2.2, 3.3),
          ('d_opp', 'opposite-bank distance (mm)', 'magma', 1.5, 8.0),
          ('leak_naive_3T', 'ribbon-averaging leakage, 3T', 'Reds', 0.0, 0.2)]
cortex = m['thickness'] > 0
fig = plt.figure(figsize=(13, 6))
for i, (k, title, cmap, lo, hi) in enumerate(panels):
    for j, (el, az) in enumerate([(0, 180), (0, 0)]):     # lateral, medial for a left hemisphere
        axx = fig.add_subplot(2, 4, i + 1 + 4 * j, projection='3d')
        v = np.where(cortex, m[k], np.nan)
        fv = np.nanmean(v[Fi], axis=1)
        col = plt.get_cmap(cmap)((np.clip(fv, lo, hi) - lo) / (hi - lo))
        col[np.isnan(fv)] = (0.85, 0.85, 0.85, 1)
        tri = axx.plot_trisurf(Vi[:, 0], Vi[:, 1], Vi[:, 2], triangles=Fi, linewidth=0, antialiased=False, shade=False)
        tri.set_facecolor(col); tri.set_edgecolor('none')
        axx.view_init(elev=el, azim=az); axx.set_axis_off(); axx.set_box_aspect(np.ptp(Vi, axis=0))
        if j == 0:
            axx.set_title(title, fontsize=9)
        if j == 1:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)); sm.set_array([])
            fig.colorbar(sm, ax=axx, orientation='horizontal', fraction=0.05, pad=0.02)
fig.suptitle(f'Subject {S}, left hemisphere (very inflated; top lateral, bottom medial)', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'figW3_example_maps.png'), dpi=150); plt.close()

# ---------------- report
def fmt(k, p=2):
    return f'{G[k][0]:.{p}f} ± {G[k][1]:.{p}f}'


lines = ['# MVE week 1 — resolvability and bank-proximity maps on 20 HCP subjects', '',
         f'Subjects: {len(subs)} (MVE calibration families), both hemispheres; surfaces warped MNI→ACPC with each subject\'s HCP warps.', '',
         '## Warp sanity check',
         f'- median |pial−white| / FreeSurfer thickness: ACPC {fmt("pw_thick_acpc", 3)} vs MNI {fmt("pw_thick_mni", 3)}',
         f'- correlation with thickness: ACPC {fmt("r_acpc", 3)} vs MNI {fmt("r_mni", 3)}',
         f'- midthickness area per hemisphere (ACPC): {fmt("area_cm2", 0)} cm²', '',
         '## Resolvable isotropic tangential wavelength (area-weighted medians, mean ± SD over hemispheres)',
         '| | idealised formula | ' + ' | '.join(CFG_LAB[c] for c in CFGS) + ' |', '|---|---|---|---|---|']
for acq in ACQS:
    lines.append(f'| {acq} | {fmt(f"lam_ideal_{acq}_med")} mm | ' + ' | '.join(f'{fmt(f"lam_{acq}_{c}_med")} mm' for c in CFGS) + ' |')
lines += [f'| fsaverage5 (idealised) | 3T {wquant(fs5["3T"], w5, 0.5):.2f} mm, 7T {wquant(fs5["7T"], w5, 0.5):.2f} mm | | | |', '',
          '## Degrees of freedom for signal per hemisphere (lookup-based)',
          '| | ' + ' | '.join(CFG_LAB[c] for c in CFGS) + ' |', '|---|---|---|---|']
for acq in ACQS:
    lines.append(f'| {acq} | ' + ' | '.join(f'{G[f"dfs_{acq}_{c}"][0]:,.0f} ± {G[f"dfs_{acq}_{c}"][1]:,.0f}' for c in CFGS) + ' |')
lines += ['', '## Sulcal-bank proximity and leakage',
          f'- median distance to nearest opposite bank: {fmt("dopp_median")} mm',
          f'- cortex with an opposite bank within 3 / 4 / 5 / 6 mm: ' + ' / '.join(f'{100*G[f"frac_dopp_lt{t}"][0]:.1f}%' for t in (3, 4, 5, 6)),
          f'- predicted mean cross-bank leakage of ribbon averaging: 3T {100*G["leak_naive_3T_mean"][0]:.2f}%, 7T {100*G["leak_naive_7T_mean"][0]:.2f}%',
          '- cortex where bank-specific patterns are NOT recoverable even for the optimal estimator (transfer < 0.5):',
          '  | pattern wavelength | 3T CNR 1 | 3T CNR 10 | 7T CNR 1 | 7T CNR 10 |', '  |---|---|---|---|---|'] + [
          f'  | {lab} | ' + ' | '.join(f'{100*G[f"unrec_{tag}_{acq}_cnr{c}"][0]:.1f}%' for acq in ACQS for c in (1, 10)) + ' |'
          for tag, lab in (('tau8', '7.9 mm'), ('tau5', '5.2 mm'), ('tau4', '3.9 mm'))] + ['',
          'Figures: figW1_resolvable_wavelength.png, figW2_banks.png, figW3_example_maps.png. Tables: week1_per_hemisphere.csv, week1_group_mean_sd.json.',
          '', 'Standard mapping modelled as box-voxel overlap averaging (as in HCP ribbon mapping), averaged over 6 random lattice positions.',
          'Assumptions: acquisition axes = ACPC axes (AutoAlign); planar-patch lookups (curvature ignored); BOLD FWHM 3.0 mm (3T) / 1.5 mm (7T);',
          'T2* 45 ms (3T) / 28 ms (7T); partial Fourier ignored; opposite-bank test uses a 15 mm geodesic threshold judged on the registration sphere.']
open(os.path.join(OUT, 'week1_report.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
