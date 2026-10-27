"""Week 16, local summaries over subjects.
  psf    IN_DIR OUT_PREFIX          B: residual against the PSF stretch factor (variant_eval files of the tuning subjects)
  thr    IN_DIR OUT_PREFIX          G1: leakage figures at far thresholds 10 / 15 / 20 mm and other-hemisphere only
  geo    IN_DIR OUT_PREFIX          G2: geodesic against sphere-proxy far criterion (v3 files: calibrated threshold, gross, pair statistics)
  leakpsf IN_DIR OUT_PREFIX         B follow-up: leakage maps with the kernels narrowed to the scan optimum
  f2     IN_DIR F_DIR OUT_PREFIX    F2 (addendum): correlated likelihood re-tuned on the tuning subjects, compared on the development subjects
  corr   IN_DIR OUT_PREFIX          F: correlated-noise against white-noise fits (variant_eval files of the development subjects)
  h3     FULL_DIR HALF1_DIR HALF2_DIR OUT_PREFIX MODEL_FULL COMP_FULL MODEL_HALF COMP_HALF   E2: H3 with re-tuned settings
"""
import sys, json, glob, re
import numpy as np
from scipy.stats import wilcoxon

BINK = ['bin_0-0.02', 'bin_0.02-0.05', 'bin_0.05-0.1', 'bin_0.1-0.2', 'bin_0.2-1.01']
CHOSEN = {'2.4': 'model_r12_s0.125', '3.2': 'model_r24_s0.25'}
fm = lambda x, d=4: f'{x:+.{d}f}'.replace('-', '−')
fp = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'


def stats(v, direction=None):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return dict(n=int(len(v)))
    b = np.random.default_rng(0).choice(v, size=(10000, len(v)), replace=True).mean(1)
    d = dict(n=int(len(v)), mean=float(v.mean()), n_pos=int((v > 0).sum()), n_neg=int((v < 0).sum()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    if np.any(v != 0):
        d['p_two_sided'] = float(wilcoxon(v).pvalue)
        if direction:
            d['p_one_sided'] = float(wilcoxon(v, alternative=direction).pvalue)
    return d


def load(indir, suffix):
    return [json.load(open(f)) for f in sorted(glob.glob(f'{indir}/*{suffix}'))]


def psf(indir, prefix):
    J = load(indir, '.variant_eval.json'); subs = [j['subject'] for j in J]
    L = [f'# Week 16 B: PSF-scale scan ({len(subs)} tuning subjects; chosen settings; cross-run residual, mean over subjects)', '',
         'α stretches the inversion\'s coarse-grid kernels (data and rendering unchanged); "all" = all three axes, "y" = phase-encoding axis only,',
         '"z" = slice axis only. "better than α = 1": subjects in which the stretched kernel gives the lower residual.', '']
    res = {'subjects': subs, 'grids': {}}
    for g in ('2.4', '3.2'):
        base = CHOSEN[g]; res['grids'][g] = {}
        keys = sorted({m for j in J for m in j['grids'][g] if m.startswith(base)}, key=lambda m: (len(m), m))
        L += [f'## {g} mm', '', '| α | axes | all coherent | better than α = 1 | leak > 10% | better | pericalcarine | better |', '|---|---|---|---|---|---|---|---|']
        for m in keys:
            mm = re.match(base + r'(?:_psf([0-9.]+)([xyz]*))?$', m)
            if not mm:
                continue
            alpha = float(mm.group(1)) if mm.group(1) else 1.0; axes = mm.group(2) or ('all' if mm.group(1) else '–')
            row = {}
            cells = []
            for st in ('coherent', 'clean16_leak10', 'calc'):
                v = np.array([j['grids'][g][m][st] for j in J if j['grids'][g].get(m, {}).get(st) is not None], float)
                v0 = np.array([j['grids'][g][base][st] for j in J if j['grids'][g].get(m, {}).get(st) is not None], float)
                row[st] = dict(mean=float(v.mean()), n=int(len(v)), better=int((v < v0).sum()))
                cells.append(f'{v.mean():.4f} | {int((v < v0).sum())}/{len(v)}' if m != base else f'{v.mean():.4f} | –')
            res['grids'][g][m] = row
            L.append(f'| {alpha:g} | {axes} | ' + ' | '.join(cells) + ' |')
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def thr(indir, prefix):
    J = load(indir, '.week16_leak_thresholds.json'); subs = [j['subject'] for j in J]
    L = [f'# Week 16 G1: leakage of standard mapping with the far criterion at other thresholds ({len(subs)} development subjects)', '',
         'Sphere proxy at 10 / 15 / 20 mm along the sheet (15 mm is the paper\'s convention) and "other hemisphere only"; emulated kernels; area-weighted',
         'means per hemisphere, averaged over hemispheres and subjects (± SD).', '']
    res = {'subjects': subs, 'grids': {}}
    for g in ('1.6', '2.4', '3.2'):
        res['grids'][g] = {}
        L += [f'## {g} mm', '', '| far criterion | mean far leak (%) | cortex > 5% (%) | cortex > 10% (%) | pericalcarine (%) |', '|---|---|---|---|---|']
        for v, lab in (('thr10', '> 10 mm'), ('thr15', '> 15 mm (used)'), ('thr20', '> 20 mm'), ('oh', 'other hemisphere only')):
            row = {}
            cells = []
            for k in ('leak_frac_mean', 'frac_leak_frac_gt5', 'frac_leak_frac_gt10', 'calc_leak_frac_mean'):
                x = np.array([50 * (j['grids'][g][f'{v}_L'][k] + j['grids'][g][f'{v}_R'][k]) for j in J])
                row[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1))); cells.append(f'{x.mean():.2f} ± {x.std(ddof=1):.2f}')
            res['grids'][g][v] = row
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def geo(indir, prefix):
    J = load(indir, '.week16_geodesic_check.json'); subs = [j['subject'] for j in J]
    cal = np.array([j['graph_over_exact_median'] for j in J]); thr = J[0]['threshold_mm']; lim = J[0]['limit_mm']
    L = [f'# Week 16 G2: true geodesic against the sphere-proxy far criterion ({len(subs)} tuning subjects; threshold {thr:g} mm, geodesic limit {lim:g} mm)', '',
         'Geodesic = Dijkstra on the edge graph of the native midthickness mesh (every undirected edge once), calibrated against Workbench\'s exact',
         f'geodesic (`wb_command -surface-geodesic-distance`, 60 random ROI vertices per hemisphere, targets at 5–{lim:g} mm): graph / exact median',
         f'{cal.mean():.4f} (subjects {cal.min():.4f}–{cal.max():.4f}; IQR within subjects about 1.02–1.05), so the far label is also evaluated at the calibrated graph threshold {thr * cal.mean():.2f} mm ("calibrated").',
         'Per hemisphere, averaged over hemispheres and subjects. Net = signed far weight over the row sum; gross = |A| far share. "label change": share of',
         'cortex (area-weighted) whose > 5% / > 10% far-leak label differs between the geodesic (uncalibrated) and sphere criteria; "mass flip": share',
         'of the operator\'s mass (|A|) whose far/near label differs.', '',
         '| grid | net far leak: geodesic / calibrated / sphere (%) | gross far: geodesic / sphere (%) | > 5%: geodesic / calibrated / sphere (%) | > 10%: geodesic / calibrated / sphere (%) | r | median: geodesic / sphere (%) | label change > 5% / > 10% (%) | mass flip (%) |',
         '|---|---|---|---|---|---|---|---|---|']
    res = {'subjects': subs, 'graph_over_exact_median_mean': float(cal.mean()), 'graph_over_exact_median_per_subject': cal.tolist(), 'threshold_mm': thr, 'limit_mm': lim, 'grids': {}, 'pairs': {}}
    for g in ('1.6', '2.4', '3.2'):
        m = lambda k: float(np.mean([0.5 * (j['grids'][g]['L'][k] + j['grids'][g]['R'][k]) for j in J]))
        res['grids'][g] = {k: m(k) for k in J[0]['grids'][g]['L']}
        r = res['grids'][g]
        L.append(f"| {g} mm | {100 * r['leak_frac_mean_geo']:.2f} / {100 * r['leak_frac_mean_cal']:.2f} / {100 * r['leak_frac_mean_sph']:.2f} | {100 * r['gross_far_mean_geo']:.2f} / {100 * r['gross_far_mean_sph']:.2f} | "
                 f"{100 * r['gt5_geo']:.1f} / {100 * r['gt5_cal']:.1f} / {100 * r['gt5_sph']:.1f} | {100 * r['gt10_geo']:.1f} / {100 * r['gt10_cal']:.1f} / {100 * r['gt10_sph']:.1f} | {r['corr']:.3f} | "
                 f"{100 * r['median_geo']:.2f} / {100 * r['median_sph']:.2f} | {100 * r['label_change_gt5']:.1f} / {100 * r['label_change_gt10']:.1f} | {100 * r['mass_flip_share']:.1f} |")
    L += ['', 'Pairs (vertex, unknown) of A in the same hemisphere, pooled over hemispheres and subjects, by sphere-proxy distance: mean geodesic / sphere ratio and the',
          f'share of pairs whose far label differs (geodesic > {thr:g} mm but sphere ≤ {thr:g} mm, and the reverse). Pairs beyond the geodesic limit are not in the ratio.', '',
          '| grid | sphere distance (mm) | pairs (millions) | geodesic / sphere | geodesic far, sphere near (%) | geodesic near, sphere far (%) |', '|---|---|---|---|---|---|']
    for g in ('1.6', '2.4', '3.2'):
        res['pairs'][g] = {}
        for b in J[0]['pairs'][f'{g}_L']:
            n = sum(j['pairs'][f'{g}_{h}'][b]['n'] for j in J for h in 'LR')
            if n == 0:
                continue
            ratio = sum(j['pairs'][f'{g}_{h}'][b]['n'] * (j['pairs'][f'{g}_{h}'][b]['mean_ratio_geo_over_sph'] or 0) for j in J for h in 'LR') / n
            a = sum(j['pairs'][f'{g}_{h}'][b]['geo_far_sph_near'] for j in J for h in 'LR') / n; c = sum(j['pairs'][f'{g}_{h}'][b]['geo_near_sph_far'] for j in J for h in 'LR') / n
            res['pairs'][g][b] = dict(n=n, ratio=ratio, geo_far_sph_near=a, geo_near_sph_far=c)
            L.append(f'| {g} mm | {b} | {n / 1e6:.1f} | {ratio:.3f} | {100 * a:.1f} | {100 * c:.1f} |')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def leakpsf(indir, prefix):
    J = load(indir, '.week16_leak_psf.json'); subs = [j['subject'] for j in J]
    L = [f'# Week 16 B (follow-up): leakage of standard mapping with the PSF model narrowed to the scan\'s optimum ({len(subs)} development subjects)', '',
         'α scales the inversion\'s kernels on all three axes (α = 1 is the paper\'s PSF model; the scan preferred 0.85 and 0.7); emulated coarse kernels, sphere',
         'criterion at 15 mm. FWHM = full width at half maximum of the kernel along x / y / z (mm, mean over subjects). Area-weighted means per hemisphere,',
         'averaged over hemispheres and subjects (± SD).', '']
    res = {'subjects': subs, 'grids': {}}
    for g in ('1.6', '2.4', '3.2'):
        res['grids'][g] = {}
        L += [f'## {g} mm', '', '| α | kernel FWHM x / y / z (mm) | net far leak (%) | gross far (%) | cortex > 5% (%) | cortex > 10% (%) | pericalcarine (%) |', '|---|---|---|---|---|---|---|']
        for v in ('psf1', 'psf0.85', 'psf0.7'):
            fw = np.mean([j['grids'][g][v]['psf_fwhm_mm_xyz'] for j in J], axis=0)
            row = {'fwhm_mm_xyz': fw.tolist()}; cells = []
            for k in ('leak_frac_mean', 'gross_far_frac_mean', 'frac_leak_frac_gt5', 'frac_leak_frac_gt10', 'calc_leak_frac_mean'):
                x = np.array([50 * (j['grids'][g][v]['L'][k] + j['grids'][g][v]['R'][k]) for j in J])
                row[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None); cells.append(f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x) > 1 else f'{x.mean():.2f}')
            res['grids'][g][v] = row
            L.append(f"| {v[3:]} | {fw[0]:.2f} / {fw[1]:.2f} / {fw[2]:.2f} | " + ' | '.join(cells) + ' |')
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def corr(indir, prefix):
    J = load(indir, '.variant_eval.json'); subs = [j['subject'] for j in J]
    L = [f'# Week 16 F: correlated-noise likelihood against the white-noise fit ({len(subs)} development subjects; chosen settings)', '',
         'Cross-run residual (mean over subjects); Δ = model − competitor (upsampling + 8 mm smoothing); D_far = Δ(highest bin) − Δ(0–2%).', '']
    res = {'subjects': subs, 'grids': {}}
    for g in ('2.4', '3.2'):
        w, c, comp = CHOSEN[g], CHOSEN[g] + 'corr', 'upsmooth8'
        strata = ['coherent', 'calc', 'clean16_leak10'] + BINK
        slab = {'coherent': 'all coherent', 'calc': 'pericalcarine', 'clean16_leak10': 'clean, leak > 10%', 'bin_0-0.02': '0–2%', 'bin_0.02-0.05': '2–5%', 'bin_0.05-0.1': '5–10%', 'bin_0.1-0.2': '10–20%', 'bin_0.2-1.01': '> 20%'}
        L += [f'## {g} mm', '', '| stratum | white | correlated | competitor | corr − white (mean, subjects corr better, two-sided p) |', '|---|---|---|---|---|']
        res['grids'][g] = {}
        for s in strata:
            vw = np.array([j['grids'][g][w].get(s, np.nan) if j['grids'][g][w].get(s) is not None else np.nan for j in J], float)
            vc = np.array([j['grids'][g][c].get(s, np.nan) if j['grids'][g].get(c, {}).get(s) is not None else np.nan for j in J], float)
            vk = np.array([j['grids'][g][comp].get(s, np.nan) if j['grids'][g][comp].get(s) is not None else np.nan for j in J], float)
            ok = np.isfinite(vw) & np.isfinite(vc); st = stats(list(vc[ok] - vw[ok]))
            res['grids'][g][s] = dict(white=float(np.nanmean(vw)), corr=float(np.nanmean(vc)), competitor=float(np.nanmean(vk)), diff=st)
            L.append(f"| {slab[s]} | {np.nanmean(vw):.4f} | {np.nanmean(vc):.4f} | {np.nanmean(vk):.4f} | {fm(st['mean'])}, {st['n_neg']}/{st['n']}, p {fp(st['p_two_sided'])} |" if 'mean' in st else f'| {slab[s]} | – | – | – | – |')
        def dfar(j, m):
            d = [(j['grids'][g][m][k] - j['grids'][g][comp][k]) if (j['grids'][g].get(m, {}).get(k) is not None and j['grids'][g][comp].get(k) is not None) else None for k in BINK]
            avail = [i for i, x in enumerate(d) if x is not None]
            return None if not avail or avail[-1] == 0 or d[0] is None else d[avail[-1]] - d[0]
        dw = [dfar(j, w) for j in J]; dc = [dfar(j, c) for j in J]
        sd_ = stats([b - a for a, b in zip(dw, dc) if a is not None and b is not None])
        res['grids'][g]['D_far'] = dict(white=float(np.nanmean([x for x in dw if x is not None])), corr=float(np.nanmean([x for x in dc if x is not None])), diff=sd_)
        L += ['', f"D_far: white {fm(res['grids'][g]['D_far']['white'])}, correlated {fm(res['grids'][g]['D_far']['corr'])}; corr − white {fm(sd_['mean'])} [{fm(sd_['ci95'][0])}, {fm(sd_['ci95'][1])}], p {fp(sd_['p_two_sided'])}", '']
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def h3(full_dir, h1_dir, h2_dir, prefix, mf, cf, mh, ch):
    """E2: Δ_clean and D_far with re-tuned half-run settings (mh, ch) against the full-run ones (mf, cf); variant_eval files."""
    F = {j['subject']: j for j in load(full_dir, '.variant_eval.json')}
    H = [{j['subject']: j for j in load(d, '.variant_eval.json')} for d in (h1_dir, h2_dir)]
    subs = sorted(set(F) & set(H[0]) & set(H[1]))
    L = [f'# Week 16 E2: H3 with settings re-tuned for half-run sources ({len(subs)} development subjects)', '',
         f'Full runs: model `{mf}`, competitor `{cf}` (week-3 choices). Half runs: model `{mh}`, competitor `{ch}` (chosen on the tuning subjects\' half runs, E1).',
         'Δ_clean = gain in the 0–2% far-leak bin; D_far = Δ(highest bin) − Δ(0–2%); half = mean of the two halves.', '',
         '| grid | quantity | mean | in direction | bootstrap 95% | p (one-sided) | p (two-sided) |', '|---|---|---|---|---|---|---|']
    res = {'subjects': subs, 'settings': dict(model_full=mf, comp_full=cf, model_half=mh, comp_half=ch), 'grids': {}}
    names = {g: tuple(x.split('|')[i] if '|' in x else x for x in (mf, cf, mh, ch)) for i, g in enumerate(('2.4', '3.2'))}
    for g in ('2.4', '3.2'):
        mf_, cf_, mh_, ch_ = names[g]
        def gain(j, m, c, k):
            a = j['grids'][g].get(m, {}).get(k); b = j['grids'][g].get(c, {}).get(k)
            return None if a is None or b is None else a - b
        def dfar(j, m, c):
            d = [gain(j, m, c, k) for k in BINK]; avail = [i for i, x in enumerate(d) if x is not None]
            return None if not avail or avail[-1] == 0 or d[0] is None else d[avail[-1]] - d[0]
        rows = {}
        rows['H3a′: Δ_clean(half, re-tuned) − Δ_clean(full) > 0'] = ([np.mean([gain(H[i][S], mh_, ch_, 'bin_0-0.02') for i in (0, 1)]) - gain(F[S], mf_, cf_, 'bin_0-0.02') for S in subs], 'greater')
        rows['H3a (fixed settings, for reference): Δ_clean(half, full-run settings) − Δ_clean(full) > 0'] = ([np.mean([gain(H[i][S], mf_, cf_, 'bin_0-0.02') for i in (0, 1)]) - gain(F[S], mf_, cf_, 'bin_0-0.02') for S in subs], 'greater')
        rows['H3b′: D_far(half, re-tuned) − D_far(full) (interval)'] = ([np.mean([dfar(H[i][S], mh_, ch_) for i in (0, 1)]) - dfar(F[S], mf_, cf_) for S in subs if all(dfar(H[i][S], mh_, ch_) is not None for i in (0, 1)) and dfar(F[S], mf_, cf_) is not None], None)
        rows['H3c′: Δ_clean(half, re-tuned) < 0'] = ([np.mean([gain(H[i][S], mh_, ch_, 'bin_0-0.02') for i in (0, 1)]) for S in subs], 'less')
        rows['Δ_clean(full) (reference)'] = ([gain(F[S], mf_, cf_, 'bin_0-0.02') for S in subs], 'less')
        rows['half-run model gain, all coherent (re-tuned)'] = ([np.mean([gain(H[i][S], mh_, ch_, 'coherent') for i in (0, 1)]) for S in subs], 'less')
        res['grids'][g] = {}
        for name, (vals, direction) in rows.items():
            st = stats(vals, direction); res['grids'][g][name] = st
            if 'mean' in st:
                nd = st['n_pos'] if direction == 'greater' else st['n_neg']
                L.append(f"| {g} mm | {name} | {fm(st['mean'])} | {nd}/{st['n']} | [{fm(st['ci95'][0])}, {fm(st['ci95'][1])}] | {fp(st['p_one_sided']) if 'p_one_sided' in st else '–'} | {fp(st['p_two_sided'])} |")
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


def f2(indir, f_dir, prefix):
    """F2 (addendum): the correlated-noise likelihood with its own prior settings. Selection on the tuning subjects (week-3 rule),
    comparison with the white likelihood at its week-3 choice on the development subjects; F reproduced; oracle bound."""
    tune = json.load(open('results/week3/week3_tuning.json'))['subjects']
    J = {j['subject']: j for j in load(indir, '.variant_eval.json')}; FJ = {j['subject']: j for j in load(f_dir, '.variant_eval.json')}
    T = [J[S] for S in tune if S in J]; D = [J[S] for S in sorted(J) if S not in tune]
    comp = 'upsmooth8'; strata = ['coherent', 'calc', 'clean16_leak10'] + BINK
    slab = {'coherent': 'all coherent', 'calc': 'pericalcarine', 'clean16_leak10': 'clean, leak > 10%', 'bin_0-0.02': '0–2%', 'bin_0.02-0.05': '2–5%', 'bin_0.05-0.1': '5–10%', 'bin_0.1-0.2': '10–20%', 'bin_0.2-1.01': '> 20%'}
    L = [f'# Week 16 F2: the correlated-noise likelihood with its own prior settings ({len(T)} tuning subjects for the choice, {len(D)} development subjects for the comparison)', '',
         'Plan: `PLAN_F2_ADDENDUM.md`. Correlated fits on r ∈ {6, 12, 24, 48} mm × s ∈ {1/32 … 1}; the week-3 rule (lowest mean cross-run residual over all',
         'coherent vertices of the tuning subjects) picks the correlated setting per grid; the white likelihood keeps its week-3 choice. Negative differences',
         'favour the correlated likelihood.', '']
    res = {'tuning_subjects': [j['subject'] for j in T], 'dev_subjects': [j['subject'] for j in D], 'grids': {}}
    cm = lambda j, g: sorted(m for m in j['grids'][g] if re.match(r'model_r[0-9.]+_s[0-9.]+corr$', m))
    for g in ('2.4', '3.2'):
        white = CHOSEN[g]; r = res['grids'][g] = {}
        keys = sorted(set.intersection(*[set(cm(j, g)) for j in T])) if T else []
        mt = {m: float(np.mean([j['grids'][g][m]['coherent'] for j in T])) for m in keys}
        wt = float(np.mean([j['grids'][g][white]['coherent'] for j in T])) if T else float('nan')
        L += [f'## {g} mm', '']
        if not mt:
            L += ['*(tuning subjects pending)*', '']; continue
        pick = min(mt, key=mt.get); r['choice'] = pick; r['tuning_mean'] = mt; r['white_tuning_mean'] = wt
        rr = sorted({float(re.match(r'model_r([0-9.]+)_', m).group(1)) for m in mt}); ss = sorted({float(re.match(r'.*_s([0-9.]+)corr', m).group(1)) for m in mt})
        L += [f'Tuning subjects, mean residual over all coherent vertices (white likelihood at its choice `{white}`: {wt:.4f}); choice **{pick}** ({mt[pick]:.4f}).', '',
              '| r \\ s | ' + ' | '.join(f'{x:g}' for x in ss) + ' |', '|---|' + '---|' * len(ss)]
        for rv in rr:
            L.append(f'| {rv:g} mm | ' + ' | '.join((f"**{mt[m]:.4f}**" if m == pick else f'{mt[m]:.4f}') if (m := f'model_r{rv:g}_s{x:g}corr') in mt else '–' for x in ss) + ' |')
        L.append('')
        if not D:
            L += ['*(development subjects pending)*', '']; continue
        Dk = [j for j in D if pick in j['grids'][g]]
        def dfar(j, m):
            d = [(j['grids'][g][m][k] - j['grids'][g][comp][k]) if (j['grids'][g].get(m, {}).get(k) is not None and j['grids'][g][comp].get(k) is not None) else None for k in BINK]
            avail = [i for i, x in enumerate(d) if x is not None]
            return None if not avail or avail[-1] == 0 or d[0] is None else d[avail[-1]] - d[0]
        fset = CHOSEN[g] + 'corr'
        L += [f'Development subjects (n = {len(Dk)}): residual by stratum, mean over subjects; differences paired per subject.', '',
              f'| stratum | white `{white}` | correlated, re-tuned `{pick}` | correlated at the white settings (F) | competitor `{comp}` | re-tuned − white (mean [95%], subjects corr better, p) |',
              '|---|---|---|---|---|---|']
        r['dev'] = {}
        for st_ in strata:
            vw = np.array([j['grids'][g][white].get(st_) if j['grids'][g][white].get(st_) is not None else np.nan for j in Dk], float)
            vc = np.array([j['grids'][g][pick].get(st_) if j['grids'][g][pick].get(st_) is not None else np.nan for j in Dk], float)
            vf = np.array([j['grids'][g].get(fset, {}).get(st_) if j['grids'][g].get(fset, {}).get(st_) is not None else np.nan for j in Dk], float)
            vk = np.array([j['grids'][g][comp].get(st_) if j['grids'][g][comp].get(st_) is not None else np.nan for j in Dk], float)
            ok = np.isfinite(vw) & np.isfinite(vc); st = stats(list(vc[ok] - vw[ok]))
            r['dev'][st_] = dict(white=float(np.nanmean(vw)), corr_retuned=float(np.nanmean(vc)), corr_white_settings=float(np.nanmean(vf)), competitor=float(np.nanmean(vk)), diff=st)
            L.append(f"| {slab[st_]} | {np.nanmean(vw):.4f} | {np.nanmean(vc):.4f} | {np.nanmean(vf):.4f} | {np.nanmean(vk):.4f} | " + (f"{fm(st['mean'])} [{fm(st['ci95'][0])}, {fm(st['ci95'][1])}], {st['n_neg']}/{st['n']}, p {fp(st['p_two_sided'])} |" if 'mean' in st else '– |'))
        dw = [dfar(j, white) for j in Dk]; dc = [dfar(j, pick) for j in Dk]
        sd_ = stats([b - a for a, b in zip(dw, dc) if a is not None and b is not None])
        r['D_far'] = dict(white=float(np.nanmean([x for x in dw if x is not None])), corr_retuned=float(np.nanmean([x for x in dc if x is not None])), diff=sd_)
        L += ['', f"D_far (model − competitor, highest bin − 0–2%): white {fm(r['D_far']['white'])}, correlated re-tuned {fm(r['D_far']['corr_retuned'])}; difference {fm(sd_['mean'])} [{fm(sd_['ci95'][0])}, {fm(sd_['ci95'][1])}], p {fp(sd_['p_two_sided'])}." if 'mean' in sd_ else '']
        md = {m: float(np.mean([j['grids'][g][m]['coherent'] for j in Dk if m in j['grids'][g]])) for m in set.intersection(*[set(cm(j, g)) for j in Dk])}
        orc = min(md, key=md.get); wd = float(np.mean([j['grids'][g][white]['coherent'] for j in Dk])); r['oracle'] = dict(method=orc, mean=md[orc], white=wd)
        L += [f'Oracle bound (best correlated setting chosen on these development subjects themselves): `{orc}`, {md[orc]:.4f} against white {wd:.4f} ({fm(md[orc] - wd)}).']
        rep = [abs(j['grids'][g][fset]['coherent'] - FJ[j['subject']]['grids'][g][fset]['coherent']) for j in Dk if j['subject'] in FJ and fset in j['grids'][g] and fset in FJ[j['subject']]['grids'][g]]
        if rep:
            r['F_reproduced_max_abs_diff'] = float(max(rep))
            L += [f'F reproduced inside F2 (`{fset}`, same code, warm-started differently): largest per-subject difference from the F run {max(rep):.1e} over {len(rep)} subjects.']
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'psf': lambda: psf(sys.argv[2], sys.argv[3]), 'thr': lambda: thr(sys.argv[2], sys.argv[3]), 'geo': lambda: geo(sys.argv[2], sys.argv[3]),
     'corr': lambda: corr(sys.argv[2], sys.argv[3]), 'h3': lambda: h3(*sys.argv[2:10]), 'leakpsf': lambda: leakpsf(sys.argv[2], sys.argv[3]), 'f2': lambda: f2(sys.argv[2], sys.argv[3], sys.argv[4])}[cmd]()
