"""Week 4, StudyForrest: aggregate per-subject sf_model logs into tables, figures and a report section.
Usage: python3 sf_summarize.py MODEL_DIR OUTDIR
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
L = {os.path.basename(d): json.load(open(f'{d}/{os.path.basename(d)}.sf_model_log.json')) for d in sorted(glob.glob(f'{MD}/sub-*'))
     if os.path.exists(f'{d}/{os.path.basename(d)}.sf_model_log.json')}
# pairs involving a session whose final registration is unreliable (bbregister cost > 0.6, the anchoring threshold) are left out
REG_MAX = 0.6
EXCLUDED = []
for S in L:
    for p in list(L[S]['pairs']):
        bad = [k for k in p.split('->') if (L[S]['sessions'][k].get('bbr_cost') or 9) > REG_MAX]
        if bad:
            EXCLUDED.append((S, p, bad)); del L[S]['pairs'][p]
METH = ['standard', 'smoothing', 'resample_smoothing', 'model']
MLAB = {'standard': 'standard mapping (source)', 'smoothing': 'standard + smoothing (10 mm)',
        'resample_smoothing': 'resample to target + smoothing (8 mm)', 'model': 'model-based'}
MCOL = {'standard': '#999999', 'smoothing': '#8FAADC', 'resample_smoothing': '#55A868', 'model': '#C44E52'}
SETS = ['covered', 'visual', 'pericalcarine']
RES = {'r08': 0.8, 'r14': 1.4, 'r20': 2.0, 'r30': 3.0}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def pkey(p):
    s, t = p.split('->')
    return (s.startswith('3T'), -RES[s.split('-')[1]], RES[t.split('-')[1]])


PAIRS = sorted({p for S in L for p in L[S]['pairs']}, key=pkey)


def val(S, p, m, st):
    v = L[S]['pairs'].get(p, {}).get(m, {})
    return v.get(st) if isinstance(v, dict) else None


def ms(x):
    x = np.asarray([v for v in x if v is not None], float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, int(len(x))) if len(x) else (np.nan, np.nan, 0)


rows = {}
for p in PAIRS:
    subs = [S for S in L if p in L[S]['pairs']]
    r = {'subjects': subs}
    for st in SETS:
        r[st] = {m: ms([val(S, p, m, st) for S in subs]) for m in METH + ['ceiling_split_half']}
        for comp in ('smoothing', 'resample_smoothing'):
            d = [val(S, p, 'model', st) - val(S, p, comp, st) for S in subs
                 if val(S, p, 'model', st) is not None and val(S, p, comp, st) is not None]
            r[st][f'model_minus_{comp}'] = dict(mean=float(np.mean(d)) if d else np.nan, n=len(d), model_better=int(sum(x < 0 for x in d)))
        r[st]['n_vertices'] = ms([L[S]['pairs'][p]['n'][st] for S in subs])
        # exploratory FWHM curves (x_* entries; absent in logs written before they were added)
        xs = sorted({m for S in subs for m in L[S]['pairs'][p] if m.startswith('x_')})
        r[st]['explore'] = {m: ms([val(S, p, m, st) for S in subs]) for m in xs}
    rows[p] = r
json.dump(rows, open(f'{OUT}/sf_group.json', 'w'), indent=1, default=float)


# ---------------- figure: paired differences to standard + smoothing, per pair (visual vertices), dots = subjects
def strip(a, pp, getters, title, ref_label):
    x = np.arange(len(pp))
    for gi, (lab, col, fn) in enumerate(getters):
        for k, p in enumerate(pp):
            d = np.array([v for v in fn(p) if v is not None and np.isfinite(v)], float)
            if len(d):
                xo = x[k] + (gi - (len(getters) - 1) / 2) * 0.22
                a.scatter(np.full(len(d), xo) + np.random.default_rng(k * 10 + gi).uniform(-0.05, 0.05, len(d)), d, s=9, color=col,
                          alpha=0.8, label=lab if k == 0 else None)
                a.plot([xo - 0.09, xo + 0.09], [d.mean()] * 2, color='k', lw=1.2)
    a.axhline(0, color='k', lw=0.8, ls=':')
    a.set_xticks(x); a.set_xticklabels([p.replace('->', ' → ') for p in pp], rotation=35, ha='right', fontsize=7)
    a.set_ylabel(f'residual − {ref_label}\n(< 0: better)', fontsize=8); a.set_title(title, fontsize=9)


def diff_to(p, m, ref, st='visual'):
    return [val(S, p, m, st) - val(S, p, ref, st) if val(S, p, m, st) is not None and val(S, p, ref, st) is not None else None
            for S in L if p in L[S]['pairs']]


fam = (('7T-', '7T source → finer 7T target'), ('3T-', '3T source → 7T target (same people and stimulus)'))
fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), squeeze=False)
for ai, (pre, title) in enumerate(fam):
    pp = [p for p in PAIRS if p.startswith(pre)]
    strip(ax[ai, 0], pp, [('standard mapping', MCOL['standard'], lambda p: diff_to(p, 'standard', 'smoothing')),
                          ('resample + smoothing (8 mm)', MCOL['resample_smoothing'], lambda p: diff_to(p, 'resample_smoothing', 'smoothing')),
                          ('model (HCP prior)', MCOL['model'], lambda p: diff_to(p, 'model', 'smoothing'))],
          f'{title}: visual vertices, settings fixed from HCP', 'standard + smoothing (10 mm)')
ax[0, 0].legend(frameon=False, fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(f'{OUT}/figW4-3_sf_prediction.png', dpi=150); plt.close()

# ---------------- exploratory figure: FWHM curves in visual cortex (7T pairs, and 3T pairs if present)
pp = [p for p in PAIRS if rows[p]['visual']['explore']]
if pp:
    nc = min(len(pp), 6); nr = int(np.ceil(len(pp) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.7 * nr), squeeze=False)
    for k, p in enumerate(pp):
        a = ax[k // nc, k % nc]; E = rows[p]['visual']['explore']; M = rows[p]['visual']
        CLAB = {'smoothing': 'standard + smoothing', 'resample_smoothing': 'resample + smoothing',
                'model': 'model (HCP prior) + post-smoothing'}
        for base, pre, col, nominal in (('smoothing', 'x_smooth', MCOL['smoothing'], 10.0),
                                        ('resample_smoothing', 'x_resample_smooth', MCOL['resample_smoothing'], 8.0),
                                        ('model', 'x_model_smooth', MCOL['model'], 0.0)):
            pts = {float(m[len(pre):]): E[m][0] for m in E if m.startswith(pre) and m[len(pre):].replace('.', '').isdigit()}
            pts[nominal] = M[base][0]
            xs = sorted(pts); a.plot(xs, [pts[x] for x in xs], 'o-', ms=3, color=col, label=CLAB[base])
        grid = {m: E[m][0] for m in E if m.startswith('x_model_r')}
        if grid:
            a.axhline(min(grid.values()), color=MCOL['model'], lw=1, ls='--', label='model, best prior setting (mean over subjects)')
        a.axhline(M['standard'][0], color=MCOL['standard'], lw=0.8, ls=':', label='standard mapping')
        a.set_title(p.replace('->', ' → ') + f" (n={M['model'][2]})", fontsize=8); a.set_xlabel('surface smoothing FWHM (mm)', fontsize=7)
        if k % nc == 0:
            a.set_ylabel('residual, visual vertices', fontsize=7)
    for k in range(len(pp), nr * nc):
        ax[k // nc, k % nc].axis('off')
    ax[0, 0].legend(frameon=False, fontsize=6)
    plt.tight_layout(); plt.savefig(f'{OUT}/figW4-4_sf_fwhm_curves_exploratory.png', dpi=150); plt.close()

# ---------------- tables
f = lambda t, p=3: f'{t[0]:.{p}f} ± {t[1]:.{p}f}' if np.isfinite(t[0]) else '–'
T = ['## StudyForrest: predicting independently acquired finer maps (real acquisitions)', '',
     "Residual energy fraction of the target session's standard map (left- and right-hemifield betas) after the best real scaling; "
     'mean ± SD over subjects; all settings fixed in advance from the HCP tuning. "Model better" counts subjects where the model '
     'residual is below that competitor.', '']
for st in SETS:
    T += [f'### {st} vertices', '',
          '| pair | subjects | vertices | ' + ' | '.join(MLAB[m] for m in METH) + " | target's odd vs even runs | model − smoothing (better) | model − resample+smoothing (better) |",
          '|---|---|---|' + '---|' * len(METH) + '---|---|---|']
    for p in PAIRS:
        r = rows[p][st]
        if r['model'][2] == 0:
            continue
        d, e = r['model_minus_smoothing'], r['model_minus_resample_smoothing']
        T.append(f"| {p.replace('->', ' → ')} | {r['model'][2]} | {r['n_vertices'][0]:.0f} | " + ' | '.join(f(r[m]) for m in METH) +
                 f" | {f(r['ceiling_split_half'])} | {d['mean']:+.3f} ({d['model_better']}/{d['n']}) | {e['mean']:+.3f} ({e['model_better']}/{e['n']}) |")
    T.append('')
T += ['Pooled over pairs and subjects (pre-specified settings): model − smoothing / model − resample+smoothing, mean '
      '(subject-pairs where the model is better):', '']
for st in ('covered', 'visual'):
    for fam_, pre in (('7T→7T', '7T-'), ('3T→7T', '3T-'), ('all', '')):
        d1 = [val(S, p, 'model', st) - val(S, p, 'smoothing', st) for p in PAIRS if p.startswith(pre) for S in L
              if val(S, p, 'model', st) is not None and val(S, p, 'smoothing', st) is not None]
        d2 = [val(S, p, 'model', st) - val(S, p, 'resample_smoothing', st) for p in PAIRS if p.startswith(pre) for S in L
              if val(S, p, 'model', st) is not None and val(S, p, 'resample_smoothing', st) is not None]
        if d1:
            d1, d2 = np.array(d1), np.array(d2)
            T.append(f'- {fam_}, {st}: {d1.mean():+.3f} ({int((d1 < 0).sum())}/{len(d1)}) / {d2.mean():+.3f} ({int((d2 < 0).sum())}/{len(d2)})')
T.append('')
# exploratory: best FWHM per method, visual vertices
if pp:
    T += ['### Exploratory (not pre-specified): smoothing-width curves, visual vertices', '',
          '| pair | smoothing: best FWHM → residual (10 mm: nominal) | resample + smoothing: best FWHM → residual (8 mm) | model post-smoothed: best FWHM → residual (0 = nominal) |',
          '|---|---|---|---|']
    for p in pp:
        E = rows[p]['visual']['explore']; M = rows[p]['visual']; cells = []
        for base, pre, nominal in (('smoothing', 'x_smooth', 10.0), ('resample_smoothing', 'x_resample_smooth', 8.0), ('model', 'x_model_smooth', 0.0)):
            pts = {float(m[len(pre):]): E[m][0] for m in E if m.startswith(pre) and m[len(pre):].replace('.', '').isdigit()}
            pts[nominal] = M[base][0]; b = min(pts, key=pts.get)
            cells.append(f'{b:g} mm → {pts[b]:.3f} ({nominal:g} mm: {pts[nominal]:.3f})')
        T.append(f"| {p.replace('->', ' → ')} | " + ' | '.join(cells) + ' |')
    T.append('')
# exploratory: settings re-tuned on these data, leave-one-subject-out (each held-out subject scored with the setting that is best
# on the other subjects for the same pair and vertex set)
FAMS = {'smoothing': lambda m: m == 'smoothing' or (m.startswith('x_smooth') and m[8:].replace('.', '').isdigit()),
        'resample_smoothing': lambda m: m == 'resample_smoothing' or m.startswith('x_resample_smooth'),
        'model': lambda m: m.startswith('x_model_r')}


def loso(p, st, fam):
    # subjects that have every family for this pair (the prior grid exists only in --prior-grid runs)
    subs = [S for S in L if p in L[S]['pairs'] and all(any(FAMS[f](m) for m in L[S]['pairs'][p]) for f in FAMS)]
    keys = sorted({m for S in subs for m in L[S]['pairs'][p] if FAMS[fam](m)})
    if not keys or len(subs) < 3:
        return None
    M = np.array([[val(S, p, m, st) if val(S, p, m, st) is not None else np.nan for m in keys] for S in subs], float)
    if np.isnan(M).all(1).any():
        return None
    out, chosen = [], []
    for i in range(len(subs)):
        tr = np.delete(M, i, 0); j = int(np.nanargmin(np.nanmean(tr, 0))); out.append(M[i, j]); chosen.append(keys[j])
    return dict(scores=out, chosen=chosen, subjects=subs, oracle=keys[int(np.nanargmin(np.nanmean(M, 0)))])


LO = {}
for p in PAIRS:
    for st in ('covered', 'visual'):
        r = {fam: loso(p, st, fam) for fam in FAMS}
        if all(r.values()):
            LO[f'{p}|{st}'] = r
if LO:
    json.dump(LO, open(f'{OUT}/sf_loso_exploratory.json', 'w'), indent=1, default=float)
    # figure: re-tuned (leave-one-subject-out) differences to re-tuned smoothing, visual vertices
    kk = [k for k in LO if k.endswith('|visual')]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), squeeze=False)
    for ai, (pre, title) in enumerate(fam):
        pk = [k for k in kk if k.startswith(pre)]
        if not pk:
            ax[ai, 0].axis('off'); continue
        get = lambda f: (lambda p: list(np.array(LO[p]['%s' % f]['scores']) - np.array(LO[p]['smoothing']['scores'])))
        strip(ax[ai, 0], pk, [('resample + smoothing, re-tuned', MCOL['resample_smoothing'], get('resample_smoothing')),
                              ('model, re-tuned', MCOL['model'], get('model'))],
              f'{title}: visual vertices, every method re-tuned (leave-one-subject-out; exploratory)', 'smoothing, re-tuned')
        ax[ai, 0].set_xticklabels([k.split('|')[0].replace('->', ' → ') for k in pk], rotation=35, ha='right', fontsize=7)
    ax[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(f'{OUT}/figW4-5_sf_retuned_exploratory.png', dpi=150); plt.close()
    T += ['### Exploratory (not pre-specified): every method re-tuned on StudyForrest, leave-one-subject-out', '',
          'Smoothing FWHM 2–16 mm; model prior range 3 / 6 / 12 / 24 mm × s.d. scale 0.125–2. Each held-out subject is scored with the setting '
          'that is best on the other subjects (same pair and vertex set).', '',
          '| pair | vertices | standard + smoothing | resample + smoothing | model | model better than smoothing / resample+smoothing | settings chosen most often (smoothing; resample; model) |',
          '|---|---|---|---|---|---|---|']
    pooled = {}
    for k, r in LO.items():
        p, st = k.split('|')
        sm, rs, mo = (np.array(r[f]['scores']) for f in ('smoothing', 'resample_smoothing', 'model'))
        fam_ = '3T→7T' if p.startswith('3T') else '7T→7T'
        for key_, d1, d2 in (((fam_, st), mo - sm, mo - rs), (('all', st), mo - sm, mo - rs)):
            pooled.setdefault(key_, [[], []]); pooled[key_][0] += list(d1); pooled[key_][1] += list(d2)
        NICE = {'smoothing': 'smooth10', 'resample_smoothing': 'resample_smooth8'}
        top = lambda c: sorted(set(c), key=lambda k: (-c.count(k), k))[0]          # most frequent; ties broken alphabetically
        mode = lambda c: NICE.get(top(c), top(c)).replace('x_', '').replace('model_', '')
        T.append(f"| {p.replace('->', ' → ')} | {st} | {sm.mean():.3f} ± {sm.std(ddof=1):.3f} | {rs.mean():.3f} ± {rs.std(ddof=1):.3f} | "
                 f"{mo.mean():.3f} ± {mo.std(ddof=1):.3f} | {int((mo < sm).sum())}/{len(mo)} / {int((mo < rs).sum())}/{len(mo)} | "
                 f"{mode(r['smoothing']['chosen'])}; {mode(r['resample_smoothing']['chosen'])}; {mode(r['model']['chosen'])} |")
    T += ['', 'Pooled over pairs and subjects (re-tuned, leave-one-subject-out): model − smoothing / model − resample+smoothing, mean '
          '(subject-pairs where the model is better):', '']
    for (fam_, st), (d1, d2) in sorted(pooled.items()):
        d1, d2 = np.array(d1), np.array(d2)
        T.append(f'- {fam_}, {st}: {d1.mean():+.3f} ({int((d1 < 0).sum())}/{len(d1)}) / {d2.mean():+.3f} ({int((d2 < 0).sum())}/{len(d2)})')
    T.append('')
sess = {S: L[S]['sessions'] for S in L}
T += ['### Registration (bbregister cost after within-subject anchoring; lower is better)', '',
      '| subject | ' + ' | '.join(k for k in ('7T-r08', '7T-r14', '7T-r20', '7T-r30', '3T-r14', '3T-r20', '3T-r30')) + ' |', '|---|' + '---|' * 7]
for S, s in sess.items():
    T.append(f'| {S} | ' + ' | '.join((f"{s[k]['bbr_cost']:.2f}" + (' (a)' if str(s[k].get('bbr_start')).startswith('anchor') else ''))
                                      if k in s and s[k].get('bbr_cost') is not None else '–'
                                      for k in ('7T-r08', '7T-r14', '7T-r20', '7T-r30', '3T-r14', '3T-r20', '3T-r30')) + ' |')
T += ['', '(a) = improved by anchoring to the subject\'s best-registered session.', '']
if EXCLUDED:
    T += [f'Excluded (registration cost > {REG_MAX}): ' + '; '.join(f"{S} {p.replace('->', ' → ')} ({', '.join(b)})" for S, p, b in EXCLUDED), '']
open(f'{OUT}/sf_report_section.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
