"""Week 5 summary: Part A (within-session emulation on StudyForrest) and Part B (source-only self-tuning of the week-4
cross-session pairs). Three ways of choosing each method's setting:
  HCP     fixed from the HCP week-3 tuning (smoothing 10 mm, upsampling + smoothing 8 mm, model by coarsening factor)
  self    source-only split-half choice (no target data; per source and vertex set)
  LOSO    best on the other subjects (same session / factor / pair and vertex set)
Usage: python3 summarize_w5.py EMUL_DIR SELFTUNE_DIR W4_LOG_DIR OUTDIR
"""
import os, sys, json, glob
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ED, SD, W4, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
HCP_MODEL = {'1.5': 'model_r12_s0.125', '2': 'model_r24_s0.25'}
FW = ('2', '3', '4', '6', '8', '10', '12', '16')
SETS = ('covered', 'visual')
SETS_A = ('covered', 'visual', 'visual_xhemi')      # Part A adds visual vertices mixed across hemispheres at the coarse grid
MIN_N = 100                                         # a subject enters a vertex set's table only with >= this many vertices
COL = {'standard': '#999999', 'smoothing': '#8FAADC', 'upsample': '#55A868', 'model': '#C44E52', 'native': '#000000'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fam_keys = {'smoothing': [f'smooth{f}' for f in FW], 'upsample': ['upsample'] + [f'upsmooth{f}' for f in FW],
            'model': None, 'native': ['native'] + [f'native_smooth{f}' for f in FW]}
HCP_PICK = {'smoothing': lambda fac: 'smooth10', 'upsample': lambda fac: 'upsmooth8', 'model': lambda fac: HCP_MODEL[fac]}


def corr_fw(k):
    """Split-half choice of a smoothing FWHM -> full-data FWHM. Heat-equation smoothing (C + tG)^-1 C is the posterior mean of a
    Gaussian prior under white noise with t proportional to the noise variance; a half has ~2x the noise of the full data, so
    FWHM (proportional to sqrt t) shrinks by sqrt 2. Returns the nearest grid key of the same family."""
    pre = k.rstrip('0123456789.')
    if not k[len(pre):]:
        return k
    t = float(k[len(pre):]) / np.sqrt(2)
    return pre + min(FW, key=lambda g: abs(np.log(float(g) / t)))


def ms(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, int(len(x))) if len(x) else (np.nan, np.nan, 0)


def f3(t):
    return f'{t[0]:.3f} ± {t[1]:.3f}' if np.isfinite(t[0]) else '–'


# ============================================================================ Part A
A = {}
for p in sorted(glob.glob(f'{ED}/sub-*/*.w5_emul_log.json')):
    L = json.load(open(p))
    if 'factors' in L and L['factors']:
        A[(L['subject'], L['session'])] = L
T = ['# Week 5 tables', '', '## Part A: within-session emulation (StudyForrest 7T; coarse half → other half at native resolution)', '']
partA = {}
if A:
    cells = sorted({(k[1], fac) for k, L in A.items() for fac in L['factors']})
    for key, fac in cells:
        subs_all = sorted(S for (S, k), L in A.items() if k == key and fac in L['factors'] and len(L['factors'][fac]['directions']) == 2)
        if not subs_all:
            continue
        D_all = {S: A[(S, key)]['factors'][fac]['directions'] for S in subs_all}
        mkeys = sorted(k for k in next(iter(D_all.values()))['A->B']['target'] if k.startswith('model_r'))
        fams = dict(fam_keys, model=mkeys)
        res = {}
        for st in SETS_A:
            subs = [S for S in subs_all if all(D_all[S][d]['n'].get(st, 0) >= MIN_N for d in D_all[S])]
            if len(subs) < 3:
                continue
            D = {S: D_all[S] for S in subs}
            tavg = lambda S, k: np.mean([D[S][d]['target'][k][st] for d in D[S]])        # direction-averaged target score
            r = {'standard': [tavg(S, 'standard') for S in subs]}
            for fam, ks in fams.items():
                # LOSO over subjects on direction-averaged scores
                M = np.array([[tavg(S, k) for k in ks] for S in subs])
                lo = []
                for i in range(len(subs)):
                    tr = np.delete(M, i, 0) if len(subs) > 1 else M
                    lo.append(M[i, int(np.argmin(tr.mean(0)))])
                r[f'{fam}_loso'] = lo
                if fam != 'native':
                    r[f'{fam}_hcp'] = [tavg(S, HCP_PICK[fam](fac)) for S in subs]
                    # self-tuned: per direction, the source-only split-half choice
                    sel, chosen = [], []
                    for S in subs:
                        v = []
                        for d in D[S]:
                            k = min(ks, key=lambda k: D[S][d]['cv'][k][st]); v.append(D[S][d]['target'][k][st]); chosen.append(k)
                        sel.append(np.mean(v))
                    r[f'{fam}_self'] = sel; r[f'{fam}_self_chosen'] = Counter(chosen).most_common(3)
                    if fam in ('smoothing', 'upsample'):          # competitors: split-half FWHM corrected to full-data noise (primary)
                        r[f'{fam}_selfraw'] = sel
                        r[f'{fam}_self'] = [np.mean([D[S][d]['target'][corr_fw(min(ks, key=lambda k: D[S][d]['cv'][k][st]))][st]
                                                     for d in D[S]]) for S in subs]
            r['native_raw'] = [tavg(S, 'native') for S in subs]
            r['calibration'] = [np.mean([D[S][d]['noise_calibration'] for d in D[S]]) for S in subs]
            r['n'] = [np.mean([D[S][d]['n'][st] for d in D[S]]) for S in subs]
            r['subjects'] = subs
            res[st] = r
        partA[(key, fac)] = dict(subjects=subs_all, res=res,
                                 emul_err=max(A[(S, key)]['factors'][fac]['emulate_check_rel_err'] for S in subs_all),
                                 voxel=A[(subs_all[0], key)]['factors'][fac]['voxel_mm'])
    for st in SETS_A:
        T += [f'### {st} vertices (mean ± SD over subjects; each subject = mean of the two directions)', '',
              '| native session → coarse grid | subjects | standard | smoothing: HCP / self / LOSO | upsampling + smoothing: HCP / self / LOSO | '
              'model: HCP / self / LOSO | source half at native resolution: raw / LOSO-smoothed | model − best competitor, self (better) | same, LOSO (better) |',
              '|---|---|---|---|---|---|---|---|---|']
        for (key, fac), P in partA.items():
            if st not in P['res']:
                continue
            r = P['res'][st]; g = lambda k: ms(r[k])
            d_self = np.array(r['model_self']) - np.minimum(r['smoothing_self'], r['upsample_self'])
            d_loso = np.array(r['model_loso']) - np.minimum(r['smoothing_loso'], r['upsample_loso'])
            T.append(f"| {key} → ×{fac} ({P['voxel'][0]:.1f} mm) | {len(r['subjects'])} ({np.mean(r['n']):.0f} v) | {f3(g('standard'))} | "
                     f"{g('smoothing_hcp')[0]:.3f} / {g('smoothing_self')[0]:.3f} / {g('smoothing_loso')[0]:.3f} | "
                     f"{g('upsample_hcp')[0]:.3f} / {g('upsample_self')[0]:.3f} / {g('upsample_loso')[0]:.3f} | "
                     f"{g('model_hcp')[0]:.3f} / {g('model_self')[0]:.3f} / {g('model_loso')[0]:.3f} | "
                     f"{g('native_raw')[0]:.3f} / {g('native_loso')[0]:.3f} | "
                     f"{d_self.mean():+.4f} ({int((d_self < 0).sum())}/{len(d_self)}) | {d_loso.mean():+.4f} ({int((d_loso < 0).sum())}/{len(d_loso)}) |")
        T.append('')
    # pooled over sessions and factors
    T += ['Pooled over sessions, factors and subjects: model − best competitor (smoothing or upsampling + smoothing), same selection '
          'scheme for all methods (subject-cells where the model is better):', '']
    for st in SETS_A:
        for sch in ('hcp', 'self', 'loso'):
            parts = [np.array(P['res'][st][f'model_{sch}']) - np.minimum(P['res'][st][f'smoothing_{sch}'], P['res'][st][f'upsample_{sch}'])
                     for P in partA.values() if st in P['res']]
            if not parts:
                continue
            d = np.concatenate(parts)
            T.append(f'- {st}, {sch}: {d.mean():+.4f} ({int((d < 0).sum())}/{len(d)})')
    T += ['', 'Self-tuned choices (most frequent over subjects and directions), visual vertices:', '']
    for (key, fac), P in partA.items():
        if 'visual' not in P['res']:
            continue
        r = P['res']['visual']
        T.append(f"- {key} ×{fac}: model {r['model_self_chosen']}; smoothing {r['smoothing_self_chosen']}; upsampling {r['upsample_self_chosen']}")
    T += ['', 'Noise calibration factor (observed / propagated coarse run-to-run variance), mean over subjects: ' +
          '; '.join(f"{key} ×{fac}: {np.mean(P['res']['visual']['calibration']):.2f}" for (key, fac), P in partA.items() if 'visual' in P['res']),
          'Emulation operator check (per-axis operators vs week-2 emulate(), max relative error): ' +
          f"{max(P['emul_err'] for P in partA.values()):.1e}", '']
    json.dump({f'{k}|{f}': v for (k, f), v in partA.items()}, open(f'{OUT}/w5_partA.json', 'w'), indent=1, default=float)

    # figure: paired differences to the best competitor under each scheme; visual vertices and the cross-hemisphere subset
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1]))
    for ai, (st, ttl) in enumerate((('visual', 'visual cortex'),
                                    ('visual_xhemi', 'visual cortex mixed across hemispheres at the coarse grid'))):
        ax = axs[ai]; ck = [c for c in partA if st in partA[c]['res']]
        for ci, c in enumerate(ck):
            r = partA[c]['res'][st]
            for si, (sch, col) in enumerate((('hcp', '#E8A9AB'), ('self', '#C44E52'), ('loso', '#7A1F22'))):
                d = np.array(r[f'model_{sch}']) - np.minimum(r[f'smoothing_{sch}'], r[f'upsample_{sch}'])
                x = ci + (si - 1) * 0.25
                ax.scatter(np.full(len(d), x) + np.random.default_rng(ci * 7 + si).uniform(-0.05, 0.05, len(d)), d, s=12, color=col,
                           label={'hcp': 'HCP settings', 'self': 'source-only split-half settings',
                                  'loso': 'leave-one-subject-out settings'}[sch] if ci == 0 else None)
                ax.plot([x - 0.1, x + 0.1], [d.mean()] * 2, color='k')
        ax.axhline(0, color='k', lw=0.8, ls=':'); ax.set_xticks(range(len(ck)))
        ax.set_xticklabels([f"{k} → ×{f}\n({partA[(k, f)]['voxel'][0]:.1f} mm; n={len(partA[(k, f)]['res'][st]['subjects'])})" for k, f in ck], fontsize=8)
        ax.set_ylabel('model − best competitor (< 0: model better)'); ax.set_title(ttl, fontsize=9)
        if ai == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Within-session emulation on StudyForrest 7T: residual vs the other half at native resolution (one dot per subject)', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT}/figW5-1_within_session.png', dpi=150); plt.close()

# ============================================================================ Part B
B = {}
for p in sorted(glob.glob(f'{SD}/sub-*/*.w5_selftune_*.json')):
    L = json.load(open(p)); B.setdefault(L['subject'], {}).update(L['sources'])
T += ['## Part B: source-only self-tuning for the week-4 cross-session pairs', '']
if B:
    W = {S: json.load(open(f'{W4}/{S}/{S}.sf_model_log.json')) for S in B if os.path.exists(f'{W4}/{S}/{S}.sf_model_log.json')}
    rows = {}
    for S, L in W.items():
        for pr, v in L['pairs'].items():
            src = pr.split('->')[0]
            if src not in B[S]:
                continue
            C = B[S][src]['cv']
            for st in SETS:
                km = min((k for k in C if k.startswith('model_r')), key=lambda k: C[k][st])
                ks_raw = min((k for k in C if k.startswith('smooth')), key=lambda k: C[k][st])
                ks = corr_fw(ks_raw); fw = ks[6:]                  # split-half FWHM corrected to full-data noise (see corr_fw)
                get = lambda k: v.get(k, {}).get(st) if isinstance(v.get(k), dict) else None
                sm_key = 'smoothing' if fw == '10' else f'x_smooth{fw}'
                rs_key = 'resample_smoothing' if fw == '8' else f'x_resample_smooth{fw}'
                rows.setdefault(st, []).append(dict(S=S, pair=pr, model_self=get(f'x_{km}'), smooth_self=get(sm_key), resample_self=get(rs_key),
                                                    model_hcp=get('model'), smooth_hcp=get('smoothing'), resample_hcp=get('resample_smoothing'),
                                                    chosen_model=km, chosen_fw=fw, chosen_fw_splithalf=ks_raw[6:]))
    for st in SETS:
        R_ = [r for r in rows.get(st, []) if None not in (r['model_self'], r['smooth_self'], r['resample_self'])]
        T += [f'### {st} vertices', '', '| pairs | subject-pairs | model − smoothing, self (better) | model − resample+smoothing, self (better) | '
              'model self − model HCP | smoothing self − smoothing HCP |', '|---|---|---|---|---|---|']
        for fam, pre in (('7T → 7T', '7T-'), ('3T → 7T', '3T-'), ('all', '')):
            rr = [r for r in R_ if r['pair'].startswith(pre)]
            if not rr:
                continue
            d1 = np.array([r['model_self'] - r['smooth_self'] for r in rr]); d2 = np.array([r['model_self'] - r['resample_self'] for r in rr])
            d3 = np.array([r['model_self'] - r['model_hcp'] for r in rr]); d4 = np.array([r['smooth_self'] - r['smooth_hcp'] for r in rr])
            T.append(f'| {fam} | {len(rr)} | {d1.mean():+.4f} ({int((d1 < 0).sum())}/{len(rr)}) | {d2.mean():+.4f} ({int((d2 < 0).sum())}/{len(rr)}) | '
                     f'{d3.mean():+.4f} | {d4.mean():+.4f} |')
        T.append('')
    ch = Counter(r['chosen_model'] for r in rows.get('visual', [])); cf = Counter(r['chosen_fw'] for r in rows.get('visual', []))
    T += [f'Self-tuned model settings chosen (visual, over subject-pairs): {ch.most_common(6)}; smoothing FWHM: {cf.most_common(6)}', '']
    json.dump(rows, open(f'{OUT}/w5_partB.json', 'w'), indent=1, default=float)

open(f'{OUT}/w5_tables.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
