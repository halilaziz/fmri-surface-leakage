"""Week 7 summary: the leak-gated estimator (model where the operator predicts strong mixing, competitor elsewhere) on HCP and on
StudyForrest (within-session emulation), with the gate threshold chosen leave-one-subject-out; and the group leakage maps.
Usage (xcp_d container, on TRUBA): python3 summarize_w7.py HCP_GATE_JSON SF_GATE_DIR LEAK_SUMMARY_JSON OUTDIR
"""
import os, sys, json, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HJ, SD, LJ, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]


def resid_real(target, pred, sel):
    num = den = 0.0
    for part in (np.real, np.imag):
        z, p = part(target[sel]), part(pred[sel]); c = (p @ z) / max(p @ p, 1e-30)
        num += ((z - c * p) ** 2).sum(); den += (z ** 2).sum()
    return float(num / max(den, 1e-30))


def loso_pick(M):
    """M: subjects x options. Held-out score with the option best on the other subjects."""
    return np.array([M[i, int(np.argmin(np.delete(M, i, 0).mean(0)))] for i in range(len(M))]), \
           [int(np.argmin(np.delete(M, i, 0).mean(0))) for i in range(len(M))]


T = ['# Week 7 tables: the leak-gated estimator and group leakage maps', '']
# ---------------------------------------------------------------- HCP
H = json.load(open(HJ)); subs = sorted(H)
T += ['## HCP (emulated; week-3 fits; 15 test subjects; cross-run residual vs the other run\'s 1.6 mm map)', '',
      '| grid | vertices | competitor everywhere | model everywhere | gated, tau LOSO | gated − model everywhere (better) | gated − competitor (better) | tau chosen |',
      '|---|---|---|---|---|---|---|---|']
hres = {}
for g in ('2.4', '3.2'):
    for st in ('coherent', 'clean16_leak10', 'calc'):
        comp = np.array([H[S][g]['competitor'][st] for S in subs]); mod = np.array([H[S][g]['model'][st] for S in subs])
        M = np.array([[H[S][g]['gated'][f'{t:g}'][st] for t in TAUS] for S in subs])
        gl, ch = loso_pick(M)
        hres[(g, st)] = dict(comp=comp, mod=mod, gated=gl, curve=M.mean(0))
        T.append(f"| {g} mm | {st} | {comp.mean():.4f} | {mod.mean():.4f} | {gl.mean():.4f} | {np.mean(gl - mod):+.4f} ({int((gl < mod).sum())}/{len(gl)}) | "
                 f"{np.mean(gl - comp):+.4f} ({int((gl < comp).sum())}/{len(gl)}) | {sorted({TAUS[c] for c in ch})} |")
T.append('')
# ---------------------------------------------------------------- StudyForrest
cells = {}
for p in sorted(glob.glob(f'{SD}/sub-*/*.w7_gate.npz')):
    b = os.path.basename(p).split('.'); S, key = b[0], b[1]
    fac = re.search(r'\.x([0-9.]+?)\.w7_gate\.npz$', p).group(1)          # e.g. 1.5 (the name itself contains dots)
    cells.setdefault((key, fac), {})[S] = p
sres = {}
T += ['## StudyForrest 7T within session (coarse half → other half at native resolution; residual; mean over subjects of the two '
      'directions)', '', 'Model setting and competitor chosen leave-one-subject-out on "model everywhere" / "competitor everywhere"; the threshold tau '
      'chosen leave-one-subject-out on the gated estimator. Gate: far leak of standard mapping at the coarse grid (week 2) or the '
      'other-hemisphere share.', '',
      '| native → coarse | gate | vertices | competitor everywhere | model everywhere | gated, tau LOSO | gated − competitor (better) | gated − model everywhere (better) | tau chosen |',
      '|---|---|---|---|---|---|---|---|---|']
for (key, fac), SS in sorted(cells.items()):
    subs_c = sorted(SS)
    Z = {S: np.load(SS[S]) for S in subs_c}
    mkeys = sorted(k[3:] for k in Z[subs_c[0]].files if k.startswith('AB_') and (k[3:].startswith('corr_') or k[3:].startswith('white_')))
    for st in ('visual', 'covered'):
        def sel(z, d):
            s_ = z[f'{d}_base']
            return s_ & z['visual'] if st == 'visual' else s_
        # choose the model setting and the competitor LOSO on 'everywhere' scores
        Mm = np.array([[np.mean([resid_real(Z[S][f'{d}_target'], Z[S][f'{d}_{mk}'], sel(Z[S], d)) for d in ('AB', 'BA')]) for mk in mkeys] for S in subs_c])
        comps = ['upsample', 'upsmooth2']
        Mc = np.array([[np.mean([resid_real(Z[S][f'{d}_target'], Z[S][f'{d}_{ck}'], sel(Z[S], d)) for d in ('AB', 'BA')]) for ck in comps] for S in subs_c])
        mod_l, mch = loso_pick(Mm); comp_l, cch = loso_pick(Mc)
        for gate in ('far_c', 'xh_c'):
            G = np.zeros((len(subs_c), len(TAUS)))
            for i, S in enumerate(subs_c):
                z = Z[S]; mk = mkeys[mch[i]]; ck = comps[cch[i]]
                for j, t in enumerate(TAUS):
                    v = []
                    for d in ('AB', 'BA'):
                        m, u = z[f'{d}_{mk}'].astype(complex), z[f'{d}_{ck}'].astype(complex)
                        s_ = sel(z, d); kk = np.vdot(m[s_], u[s_]) / max(np.vdot(m[s_], m[s_]).real, 1e-30)   # model onto competitor, no target
                        p = np.where(z[gate] > t, kk * m, u)
                        v.append(resid_real(z[f'{d}_target'], p, s_))
                    G[i, j] = np.mean(v)
            gl, gch = loso_pick(G)
            nv = np.mean([sel(Z[S], 'AB').sum() for S in subs_c])
            sres[(key, fac, st, gate)] = dict(comp=comp_l, mod=mod_l, gated=gl, curve=G.mean(0), subjects=subs_c)
            T.append(f"| {key} ×{fac} | {gate} | {st} ({nv:.0f}) | {comp_l.mean():.4f} | {mod_l.mean():.4f} | {gl.mean():.4f} | "
                     f"{np.mean(gl - comp_l):+.4f} ({int((gl < comp_l).sum())}/{len(gl)}) | {np.mean(gl - mod_l):+.4f} ({int((gl < mod_l).sum())}/{len(gl)}) | "
                     f"{sorted({TAUS[c] for c in gch})} |")
T.append('')
T += ['Pooled over the four StudyForrest cells (subject-cells where the gated estimator is better):', '']
for st in ('visual', 'covered'):
    for gate in ('far_c', 'xh_c'):
        ks = [k for k in sres if k[2] == st and k[3] == gate]
        if ks:
            dg = np.concatenate([sres[k]['gated'] - sres[k]['comp'] for k in ks]); dm = np.concatenate([sres[k]['mod'] - sres[k]['comp'] for k in ks])
            T.append(f'- {st}, gate {gate}: gated − competitor {dg.mean():+.4f} ({int((dg < 0).sum())}/{len(dg)}); '
                     f'model everywhere − competitor {dm.mean():+.4f} ({int((dm < 0).sum())}/{len(dm)})')
T.append('')
# ---------------------------------------------------------------- leakage maps
if os.path.exists(LJ):
    Lm = json.load(open(LJ))
    T += ['## Group leakage maps (HCP 7T geometry, fs_LR 32k, mean over subjects)', '',
          '| grid | mean far leak (L / R) | cortex with leak > 5% | > 10% | > 20% |', '|---|---|---|---|---|']
    for g, v in Lm['grids'].items():
        T.append(f"| {g} mm | {v['L']['mean']:.3f} / {v['R']['mean']:.3f} | {100 * (v['L']['area_frac_gt5'] + v['R']['area_frac_gt5']) / 2:.0f}% | "
                 f"{100 * (v['L']['area_frac_gt10'] + v['R']['area_frac_gt10']) / 2:.0f}% | {100 * (v['L']['area_frac_gt20'] + v['R']['area_frac_gt20']) / 2:.0f}% |")
    T += ['', 'Regions with the highest leak at 3.2 mm (area-weighted mean far leak at 1.6 / 2.4 / 3.2 mm; share of the region above 10% at 3.2 mm):', '']
    for nm, v in list(Lm['regions'].items())[:12]:
        T.append(f"- {nm}: {v['1.6']['mean']:.3f} / {v['2.4']['mean']:.3f} / {v['3.2']['mean']:.3f}; {100 * v['3.2']['frac10']:.0f}%")
    T.append('')
open(f'{OUT}/w7_tables.md', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
# figure: gated residual vs threshold (relative to the competitor everywhere), HCP and StudyForrest
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
xt = range(len(TAUS) + 1)
lab = ['model\neverywhere'] + ['leak\n> 0' if t == 0 else ('competitor\neverywhere' if t > 1 else f'{100 * t:g}%') for t in TAUS]
for (g, st), r in hres.items():
    if st == 'coherent':
        ax[0].plot(xt, np.r_[r['mod'].mean(), r['curve']] - r['comp'].mean(), 'o-', label=f'HCP {g} mm')
ax[0].axhline(0, color='k', lw=0.8, ls=':'); ax[0].set_xticks(list(xt)); ax[0].set_xticklabels(lab, fontsize=7)
ax[0].set_xlabel('gate: far leak of standard mapping above'); ax[0].set_ylabel('residual − competitor everywhere'); ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title('HCP (noise-limited targets), all coherent vertices', fontsize=9)
for k, r in sres.items():
    key, fac, st, gate = k
    if st == 'visual' and gate == 'far_c':
        ax[1].plot(xt, np.r_[r['mod'].mean(), r['curve']] - r['comp'].mean(), 'o-', label=f'{key} ×{fac}')
ax[1].axhline(0, color='k', lw=0.8, ls=':'); ax[1].set_xticks(list(xt)); ax[1].set_xticklabels(lab, fontsize=7)
ax[1].set_xlabel('gate: far leak of standard mapping above'); ax[1].set_ylabel('residual − competitor everywhere'); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title('StudyForrest 7T within session (resolution-limited), visual vertices', fontsize=9)
fig.suptitle('Leak-gated estimator: model where the operator predicts strong mixing, competitor elsewhere (mean over subjects)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figW7-2_gated.png', dpi=150); plt.close()
