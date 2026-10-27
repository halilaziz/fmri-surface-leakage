"""Week 7, ds002702 within-session emulation summary: every method's setting chosen leave-one-subject-out; dose-response of the
model's gain on the far leak of standard mapping at the coarse grid; the leak-gated estimator (threshold chosen leave-one-subject-out).
Usage (xcp_d container): python3 summarize_kay.py EMUL_DIR OUTDIR
"""
import os, sys, json, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ED, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
COMP = ['upsample', 'upsmooth2', 'upsmooth4', 'smooth4', 'smooth8']


def resid_multi(T, P, sel):
    num = den = 0.0
    for j in range(T.shape[1]):
        z, p = T[sel, j], P[sel, j]; c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def loso(M):
    ch = [int(np.argmin(np.delete(M, i, 0).mean(0))) for i in range(len(M))]
    return np.array([M[i, c] for i, c in enumerate(ch)]), ch


cells = {}
for p in sorted(glob.glob(f'{ED}/sub-*/*.kay_emul.npz')):
    fac = re.search(r'\.x([0-9.]+?)\.kay_emul\.npz$', p).group(1)          # e.g. 1.3125 (the name itself contains dots)
    cells.setdefault(fac, {})[os.path.basename(p).split('.')[0]] = p
T = ['# ds002702 (Kay et al.) within-session emulation: 7T 0.8 mm, coarse half → other half at native resolution', '',
     'Six ring-condition betas, each scored with its own best scaling; mean over the two directions per subject; settings chosen '
     'leave-one-subject-out (model among correlated / white-noise priors; competitor among upsampling ± 2 / 4 mm smoothing and '
     'coarse standard map + 4 / 8 mm smoothing).', '']
allres = {}
for fac, SS in sorted(cells.items(), key=lambda kv: float(kv[0])):
    subs = sorted(SS); Z = {S: np.load(SS[S]) for S in subs}
    mk = sorted(k[3:] for k in Z[subs[0]].files if k.startswith('AB_') and (k[3:].startswith('corr_') or k[3:].startswith('white_')))
    def sets(z, d):
        base = z[f'{d}_base']; vis = base & z['visual']; clean = vis & (z['far_n'] < 0.02)
        s = {'covered': base, 'visual': vis}
        for lo, hi in BINS:
            s[f'far{int(100 * lo)}'] = clean & (z['far_c'] >= lo) & (z['far_c'] < hi)
        return s
    score = lambda S, key, st: np.mean([resid_multi(Z[S][f'{d}_target'], Z[S][f'{d}_{key}'], sets(Z[S], d)[st]) for d in ('AB', 'BA')])
    R = {}
    for st in ['covered', 'visual'] + [f'far{int(100 * lo)}' for lo, _ in BINS]:
        if any(min(sets(Z[S], d)[st].sum() for d in ('AB', 'BA')) < 30 for S in subs):
            continue
        Mc = np.array([[score(S, k, st) for k in COMP] for S in subs]); Mm = np.array([[score(S, k, st) for k in mk] for S in subs])
        cl, cch = loso(Mc); ml, mch = loso(Mm)
        nat = np.array([score(S, 'native', st) for S in subs])
        R[st] = dict(comp=cl, model=ml, native=nat, comp_choice=[COMP[c] for c in cch], model_choice=[mk[c] for c in mch],
                     n=float(np.mean([sets(Z[S], 'AB')[st].sum() for S in subs])))
    # gated estimator (visual and covered): model where far_c > tau, competitor elsewhere; model scaled onto competitor, no target
    for st in ('visual', 'covered'):
        if st not in R:
            continue
        Gm = np.zeros((len(subs), len(TAUS)))
        for i, S in enumerate(subs):
            z = Z[S]; mkey, ckey = R[st]['model_choice'][i], R[st]['comp_choice'][i]
            for j, t in enumerate(TAUS):
                v = []
                for d in ('AB', 'BA'):
                    s_ = sets(z, d)[st]; m, c = z[f'{d}_{mkey}'].astype(float), z[f'{d}_{ckey}'].astype(float)
                    k = np.array([(m[s_, q] @ c[s_, q]) / max(m[s_, q] @ m[s_, q], 1e-30) for q in range(m.shape[1])])
                    g = (z['far_c'] > t)[:, None]
                    v.append(resid_multi(z[f'{d}_target'], np.where(g, m * k, c), s_))
                Gm[i, j] = np.mean(v)
        gl, gch = loso(Gm)
        R[st].update(gated=gl, gate_curve=Gm.mean(0).tolist(), tau_choice=[TAUS[c] for c in gch])
    allres[fac] = dict(res=R, subjects=subs)
    vox = json.load(open(glob.glob(f'{ED}/{subs[0]}/*x{fac}.kay_emul_log.json')[0]))['voxel_mm']
    T += [f'## 0.8 mm → ×{fac} ({vox[0]:.1f} mm), {len(subs)} subjects', '',
          '| vertices | n | best competitor (LOSO) | model (LOSO) | gated (LOSO) | source half at 0.8 mm | model − competitor (better) | gated − competitor (better) | settings (most frequent) |',
          '|---|---|---|---|---|---|---|---|---|']
    for st, r in R.items():
        dm = r['model'] - r['comp']; dg = r['gated'] - r['comp'] if 'gated' in r else None
        mf = max(set(r['model_choice']), key=r['model_choice'].count); cf = max(set(r['comp_choice']), key=r['comp_choice'].count)
        T.append(f"| {st} | {r['n']:.0f} | {r['comp'].mean():.4f} | {r['model'].mean():.4f} | " + (f"{r['gated'].mean():.4f}" if dg is not None else '–') +
                 f" | {r['native'].mean():.4f} | {dm.mean():+.4f} ({int((dm < 0).sum())}/{len(dm)}) | " +
                 (f"{dg.mean():+.4f} ({int((dg < 0).sum())}/{len(dg)})" if dg is not None else '–') + f" | {mf}; {cf}" +
                 (f"; tau {sorted(set(r['tau_choice']))}" if 'tau_choice' in r else '') + ' |')
    T.append('')
open(f'{OUT}/kay_tables.md', 'w').write('\n'.join(T) + '\n')
json.dump({f: {st: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for st, r in A['res'].items()}
           for f, A in allres.items()}, open(f'{OUT}/kay_results.json', 'w'), indent=1, default=float)
print('\n'.join(T))
# figure: dose-response and gating curve
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for fac, A in allres.items():
    R = A['res']; bk = [f'far{int(100 * lo)}' for lo, _ in BINS if f'far{int(100 * lo)}' in R]
    ax[0].errorbar(range(len(bk)), [np.mean(R[b]['model'] - R[b]['comp']) for b in bk],
                   yerr=[np.std(R[b]['model'] - R[b]['comp']) / np.sqrt(len(R[b]['model'])) for b in bk], marker='o', capsize=2, label=f'×{fac}')
    if 'gate_curve' in R.get('visual', {}):
        ax[1].plot(range(len(TAUS) + 1), np.r_[R['visual']['model'].mean(), R['visual']['gate_curve']] - R['visual']['comp'].mean(), 'o-', label=f'×{fac}')
ax[0].axhline(0, color='k', lw=0.8, ls=':'); ax[0].set_xticks(range(len(BINS))); ax[0].set_xticklabels(['0–2%', '2–5%', '5–10%', '10–20%', '> 20%'])
ax[0].set_xlabel('far leak of standard mapping at the coarse grid (visual, clean at 0.8 mm)'); ax[0].set_ylabel('model − best competitor')
ax[0].legend(frameon=False); ax[0].set_title('dose-response (mean ± SE over subjects)', fontsize=9)
ax[1].axhline(0, color='k', lw=0.8, ls=':'); ax[1].set_xticks(range(len(TAUS) + 1))
ax[1].set_xticklabels(['model\neverywhere'] + ['leak\n> 0' if t == 0 else ('competitor\neverywhere' if t > 1 else f'{100 * t:g}%') for t in TAUS], fontsize=7)
ax[1].set_xlabel('gate: far leak above'); ax[1].set_ylabel('residual − competitor everywhere'); ax[1].legend(frameon=False)
ax[1].set_title('leak-gated estimator, visual vertices', fontsize=9)
fig.suptitle('ds002702 7T 0.8 mm rings, within-session emulation', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW7-4_kay_dose_gate.png', dpi=150); plt.close()
