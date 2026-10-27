"""Week 6 summary: within-session emulation with white vs correlated-noise likelihood, and the dose-response of the model's gain on
how much of each vertex's coarse map comes from the other hemisphere.
Selection of every method's setting: self (source-only split-half; smoothing FWHM corrected by 1/sqrt 2 as in week 5) and LOSO.
Usage: python3 summarize_w6.py LOG_DIR OUTDIR
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
HCPJ = sys.argv[3] if len(sys.argv) > 3 else None          # week-6 HCP dose-response JSON (hcp_dose_response.py)
FW = ('2', '3', '4', '6', '8', '10', '12', '16')
BINS = [('xh0', '0–2%'), ('xh2', '2–5%'), ('xh5', '5–10%'), ('xh10', '10–20%'), ('xh20', '> 20%')]
SETS = ['covered', 'visual', 'visual_xhemi'] + [b for b, _ in BINS]
MIN_N = 50
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def corr_fw(k):
    pre = k.rstrip('0123456789.')
    if not k[len(pre):] or pre not in ('smooth', 'upsmooth'):
        return k
    t = float(k[len(pre):]) / np.sqrt(2)
    return pre + min(FW, key=lambda g: abs(np.log(float(g) / t)))


# collect: cells[(session, factor)][subject] = directions dict
cells = {}
for p in sorted(glob.glob(f'{LD}/sub-*/*.w6_log.json')):
    L = json.load(open(p))
    for fac, F in L['factors'].items():
        if len(F['directions']) == 2:
            cells.setdefault((L['session'], fac), {})[L['subject']] = dict(D=F['directions'], voxel=F['voxel_mm'])
res = {}
for (key, fac), SS in sorted(cells.items()):
    subs_all = sorted(SS)
    d0 = SS[subs_all[0]]['D']['A->B']['target']
    fams = {'smoothing': [f'smooth{f}' for f in FW], 'upsample': ['upsample'] + [f'upsmooth{f}' for f in FW],
            'white': sorted(k for k in d0 if k.startswith('white_')), 'corr': sorted(k for k in d0 if k.startswith('corr_'))}
    R = {}
    for st in SETS:
        subs = [S for S in subs_all if all(SS[S]['D'][d]['n'].get(st, 0) >= MIN_N and st in SS[S]['D'][d]['target'].get('standard', {})
                                           for d in SS[S]['D'])]
        if len(subs) < 3:
            continue
        D = {S: SS[S]['D'] for S in subs}
        tavg = lambda S, k: np.mean([D[S][d]['target'][k][st] for d in D[S]])
        r = {'subjects': subs, 'n': [np.mean([D[S][d]['n'][st] for d in D[S]]) for S in subs], 'native': [tavg(S, 'native') for S in subs]}
        for fam, ks in fams.items():
            M = np.array([[tavg(S, k) for k in ks] for S in subs])
            r[f'{fam}_loso'] = [M[i, int(np.argmin(np.delete(M, i, 0).mean(0)))] for i in range(len(subs))]
            cvset = 'covered' if st == 'covered' else 'visual'
            sel = []
            for S in subs:
                v = []
                for d in D[S]:
                    k = min(ks, key=lambda k: D[S][d]['cv'][k][cvset])
                    v.append(D[S][d]['target'][corr_fw(k)][st])
                sel.append(np.mean(v))
            r[f'{fam}_self'] = sel
        R[st] = r
    res[(key, fac)] = dict(res=R, voxel=SS[subs_all[0]]['voxel'],
                           calibration=float(np.mean([SS[S]['D'][d]['noise_calibration'] for S in SS for d in SS[S]['D']])))

T = ['# Week 6 tables: within-session emulation, white vs correlated-noise likelihood', '']
for st in ['visual', 'covered', 'visual_xhemi']:
    T += [f'## {st} vertices (residual vs the other half at native resolution; mean over subjects; LOSO / self-tuned)', '',
          '| native → coarse | subjects | smoothing | upsampling (+ smoothing) | model, white noise | model, correlated noise | source half native | '
          'white − best competitor (better) | corr − best competitor (better) | corr − white (better) |', '|---|---|---|---|---|---|---|---|---|---|']
    for (key, fac), P in res.items():
        if st not in P['res']:
            continue
        r = P['res'][st]
        cells_ = []
        for sch in ('loso', 'self'):
            best = np.minimum(r[f'smoothing_{sch}'], r[f'upsample_{sch}'])
            dw = np.array(r[f'white_{sch}']) - best; dc = np.array(r[f'corr_{sch}']) - best
            dcw = np.array(r[f'corr_{sch}']) - np.array(r[f'white_{sch}'])
            cells_.append((dw, dc, dcw))
        g = lambda k: np.mean(r[k])
        fmt = lambda d: f'{d.mean():+.4f} ({int((d < 0).sum())}/{len(d)})'
        T.append(f"| {key} ×{fac} ({P['voxel'][0]:.1f} mm) | {len(r['subjects'])} | {g('smoothing_loso'):.3f} / {g('smoothing_self'):.3f} | "
                 f"{g('upsample_loso'):.3f} / {g('upsample_self'):.3f} | {g('white_loso'):.3f} / {g('white_self'):.3f} | "
                 f"{g('corr_loso'):.3f} / {g('corr_self'):.3f} | {g('native'):.3f} | {fmt(cells_[0][0])} / {fmt(cells_[1][0])} | "
                 f"{fmt(cells_[0][1])} / {fmt(cells_[1][1])} | {fmt(cells_[0][2])} / {fmt(cells_[1][2])} |")
    T.append('')
# pooled and dose-response
T += ['## Pooled over sessions, factors and subjects (model − best competitor; LOSO / self-tuned)', '']
pool = {}
for st in SETS:
    for lab in ('white', 'corr'):
        for sch in ('loso', 'self'):
            ds = [np.array(P['res'][st][f'{lab}_{sch}']) - np.minimum(P['res'][st][f'smoothing_{sch}'], P['res'][st][f'upsample_{sch}'])
                  for P in res.values() if st in P['res']]
            if ds:
                d = np.concatenate(ds); pool[(st, lab, sch)] = d
    if (st, 'corr', 'loso') in pool:
        f = lambda k: f"{pool[k].mean():+.4f} ({int((pool[k] < 0).sum())}/{len(pool[k])})"
        T.append(f"- {st}: white {f((st, 'white', 'loso'))} / {f((st, 'white', 'self'))}; corr {f((st, 'corr', 'loso'))} / {f((st, 'corr', 'self'))}")
T += ['', f"Noise calibration factor (mean): " + '; '.join(f"{k} ×{f}: {P['calibration']:.2f}" for (k, f), P in res.items()), '']
open(f'{OUT}/w6_tables.md', 'w').write('\n'.join(T) + '\n')
json.dump({f'{k}|{f}': v for (k, f), v in res.items()}, open(f'{OUT}/w6_results.json', 'w'), indent=1, default=float)
print('\n'.join(T))

# figure: dose-response of the model's gain on mixing. HCP (week-3 fits, far leak incl. opposite banks) and StudyForrest
# (week-6 fits, other-hemisphere share), LOSO settings
H = json.load(open(HCPJ)) if HCPJ and os.path.exists(HCPJ) else None
fig, ax = plt.subplots(1, 3 if H else 2, figsize=(18 if H else 13, 4.2))
panels = []
if H:
    hb = ['0-0.02', '0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2-1.01']
    a = ax[0]
    for g, col, off in (('2.4', '#C44E52', -0.12), ('3.2', '#7A1F22', 0.12)):
        for bi, b in enumerate(hb):
            d = np.array([v[g][b]['model'] - min(v[g][b]['upsample_smoothing'], v[g][b]['smoothing']) for v in H.values() if v[g][b]['model'] is not None])
            a.scatter(np.full(len(d), bi + off) + np.random.default_rng(bi).uniform(-0.04, 0.04, len(d)), d, s=9, color=col, alpha=0.7,
                      label=f'{g} mm' if bi == 0 else None)
            a.plot([bi + off - 0.08, bi + off + 0.08], [d.mean()] * 2, color='k')
    a.axhline(0, color='k', lw=0.8, ls=':'); a.set_xticks(range(5)); a.set_xticklabels([bl for _, bl in BINS])
    a.set_xlabel('far leak of standard mapping at the coarse grid\n(coherent vertices clean at 1.6 mm)')
    a.set_ylabel('model − best competitor (< 0: model better)'); a.set_title('HCP 7T retinotopy, emulated 2.4 / 3.2 mm (week-3 fits)', fontsize=9)
    a.legend(frameon=False, fontsize=7)
for ai, sch in enumerate(('loso', 'self')):
    a = ax[ai + (1 if H else 0)]
    for lab, col, off in (('white', '#E8A9AB', -0.12), ('corr', '#C44E52', 0.12)):
        for bi, (b, bl) in enumerate(BINS):
            if (b, lab, sch) in pool:
                d = pool[(b, lab, sch)]
                a.scatter(np.full(len(d), bi + off) + np.random.default_rng(bi).uniform(-0.04, 0.04, len(d)), d, s=8, color=col, alpha=0.7,
                          label=f'model, {"white" if lab == "white" else "correlated"} noise' if bi == 0 else None)
                a.plot([bi + off - 0.08, bi + off + 0.08], [d.mean()] * 2, color='k')
    a.axhline(0, color='k', lw=0.8, ls=':'); a.set_xticks(range(len(BINS))); a.set_xticklabels([bl for _, bl in BINS])
    a.set_xlabel("share of the coarse map from the other hemisphere\n(visual vertices clean at native)")
    a.set_ylabel('model − best competitor (< 0: model better)')
    a.set_title(f'StudyForrest 7T, within session; settings {"leave-one-subject-out" if sch == "loso" else "source-only split-half"}', fontsize=9)
    a.legend(frameon=False, fontsize=7)
fig.suptitle("Dose-response: the model's gain grows with the amount of mixing (dots: subjects, or subject × session × factor)", fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW6-1_dose_response.png', dpi=150); plt.close()
