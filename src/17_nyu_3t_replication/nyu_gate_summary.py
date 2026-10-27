"""Week 17b summaries (local), as PLAN_NYU_GATE.md (A: coarse-data-only rule; B: registration and the pericalcarine shortfall).
  python3 src/17_nyu_3t_replication/nyu_gate_summary.py [IN_DIR]   (default results/week17/emul_gate)
Reads the rerun's score JSONs (nyu_emul.py --gate, chosen settings), the tuning choice of PLAN_NYU.md (nyu_tuning.json) and the
published week-17 scores (results/week17/emul, reproducibility check). Writes results/week17/nyu_gate.md / .json.
"""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('GEO_PROJECT', os.path.join(HERE, '..', '..'))   # folder holding results/week17
sys.path.insert(0, HERE)
from nyu_summary import load, stats, holm, dcontrast, fm, fp, SPLIT, GRIDS, BN

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results/week17/emul_gate')
OLD = os.environ.get('NYU_PUBLISHED', os.path.join(ROOT, 'results/week17/emul'))      # the published week-17 test scores
OUTP = os.path.join(ROOT, 'results/week17/nyu_gate')
T = json.load(open(os.path.join(ROOT, 'results/week17/nyu_tuning.json')))
TAUS = ['0', '0.02', '0.05', '0.1', '0.2', 'inf']


def gate_key(tau, comp):
    return comp if tau == 'inf' else f'gate_t{tau}'


def resp(J, S, g, m, s='responsive'):
    return J[S]['grids'][g]['scores'].get(m, {}).get(s)


def diff(J, subs, g, a, b, s='responsive'):
    return [(resp(J, S, g, a, s) - resp(J, S, g, b, s)) if resp(J, S, g, a, s) is not None and resp(J, S, g, b, s) is not None else None for S in subs]


def fr(rho, p):
    return f"{rho:+.2f}".replace('-', '−') + f" ({p:.2f})"


def share(J, S, g, tau):
    if tau == 'inf':
        return 0.0
    v = [J[S]['grids'][g]['info'].get(d + '_gate_share', {}).get(f't{tau}') for d in ('AB', 'BA')]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def main():
    Jt = load(IN, SPLIT['tuning']); Jx = load(IN, SPLIT['test'])
    res = dict(tuning=sorted(Jt), test=sorted(Jx), grids={}, registration={}, reproducibility={})
    L = [f'# Week 17b: coarse-data-only rule and registration (PLAN_NYU_GATE.md) — {len(Jt)} tuning, {len(Jx)} test subjects', '',
         'Δ = residual(first) − residual(second) over responsive vertices, negative favours the first; one-sided Wilcoxon (less),',
         'bootstrap 95% interval of the mean (10,000 resamples, seed 0); Holm over the four pre-specified tests (G1, G2 × two grids).', '']
    # ---- reproducibility of the published scores (same settings, same code path for model and competitor)
    Jo = load(OLD, SPLIT['test'])
    for g in GRIDS:
        model, comp = T['grids'][g]['model'], T['grids'][g]['competitor']
        dd = [abs(resp(Jx, S, g, m) - resp(Jo, S, g, m)) for S in sorted(set(Jx) & set(Jo)) for m in (model, comp, 'standard', 'upsample')
              if resp(Jx, S, g, m) is not None and resp(Jo, S, g, m) is not None]
        res['reproducibility'][g] = dict(n_values=len(dd), max_abs_diff=float(max(dd)) if dd else None)
    rep_ = '; '.join(f"{g} mm max |difference| {res['reproducibility'][g]['max_abs_diff']:.2e} over {res['reproducibility'][g]['n_values']} values"
                     for g in GRIDS if res['reproducibility'][g]['max_abs_diff'] is not None)
    L += ['Reproducibility of the published week-17 scores (model, competitor, standard mapping, upsampling; test subjects): ' + (rep_ or 'not available (no published scores found)') + '.', '']
    # ---- A. choice of tau on the tuning subjects
    L += ['## A. Gated estimator: model where CWL-P > τ, competitor elsewhere', '', '### Choice of τ (tuning subjects, mean residual over responsive vertices)', '',
          '| grid | ' + ' | '.join(f'τ = {t}' if t != 'inf' else 'τ = ∞ (competitor)' for t in TAUS) + ' | model alone |', '|---|' + '---|' * (len(TAUS) + 1)]
    prim = {}
    for g in GRIDS:
        model, comp = T['grids'][g]['model'], T['grids'][g]['competitor']; ts = sorted(Jt)
        mean = {t: float(np.mean([resp(Jt, S, g, gate_key(t, comp)) for S in ts])) for t in TAUS}
        sh = {t: float(np.mean([share(Jt, S, g, t) for S in ts])) for t in TAUS}
        tau = min(TAUS, key=lambda t: mean[t])
        mmod = float(np.mean([resp(Jt, S, g, model) for S in ts]))
        L.append(f'| {g} mm | ' + ' | '.join(('**' if t == tau else '') + f'{mean[t]:.4f}' + ('**' if t == tau else '') + f' ({100 * sh[t]:.0f}%)' for t in TAUS) + f' | {mmod:.4f} |')
        res['grids'][g] = dict(model=model, competitor=comp, tau=tau, tuning_mean=mean, tuning_share_model=sh, tuning_model_mean=mmod)
    L += ['', '(in brackets: share of responsive vertices sent to the model; bold: the chosen τ)', '',
          '### Tests (test subjects)', '', '| grid | test | τ | mean | subjects in direction | bootstrap 95% | p (one-sided) | p (Holm, 4) |', '|---|---|---|---|---|---|---|---|']
    for g in GRIDS:
        r = res['grids'][g]; gk = gate_key(r['tau'], r['competitor']); xs = sorted(Jx)
        r['G1'] = dict(stats(diff(Jx, xs, g, gk, r['competitor'])), values=diff(Jx, xs, g, gk, r['competitor']))
        r['G2'] = dict(stats(diff(Jx, xs, g, gk, r['model'])), values=diff(Jx, xs, g, gk, r['model']))
        for k_ in ('G1', 'G2'):                # a contrast that is zero for every subject (gated = competitor or = model) cannot reject:
            if 'p' not in r[k_]:               # p = 1, kept in the Holm family of four as the plan says
                r[k_]['p'] = 1.0; r[k_]['degenerate'] = True
            prim[f'{g}|{k_}'] = r[k_]['p']
    hp = holm(prim)
    for g in GRIDS:
        r = res['grids'][g]
        for k, lab in (('G1', 'G1: gated − competitor'), ('G2', 'G2: gated − model')):
            st_ = r[k]; st_['p_holm'] = hp.get(f'{g}|{k}')
            L.append(f"| {g} mm | {lab} | {r['tau']} | {fm(st_.get('mean'))} | {st_.get('n_neg', '–')}/{st_['n']} | "
                     f"[{fm(st_['ci95'][0]) if 'ci95' in st_ else '–'}, {fm(st_['ci95'][1]) if 'ci95' in st_ else '–'}] | {fp(st_.get('p'))} | {fp(st_['p_holm'])} |")
    # ---- descriptive
    L += ['', '### Descriptive (test subjects)', '', '| grid | share sent to the model | D_CWL-P (model − competitor) | D_CWL-M (for comparison) | '
          'polar-angle error: gated / model / competitor (°) | gated − model at every τ (mean) |', '|---|---|---|---|---|---|']
    for g in GRIDS:
        r = res['grids'][g]; xs = sorted(Jx); model, comp = r['model'], r['competitor']; gk = gate_key(r['tau'], comp)
        r['share_model'] = stats([share(Jx, S, g, r['tau']) for S in xs], alt='two-sided')
        r['D_cwlP'] = stats([dcontrast(Jx[S]['grids'][g]['scores'], model, comp, 'cwlP')[0] for S in xs])
        r['D_cwlM'] = stats([dcontrast(Jx[S]['grids'][g]['scores'], model, comp, 'cwlM')[0] for S in xs])
        ang = {m: float(np.mean([Jx[S]['grids'][g]['angle_deg'][m] for S in xs if m in Jx[S]['grids'][g]['angle_deg']])) for m in (gk, model, comp)}
        r['angle_deg'] = dict(gated=ang[gk], model=ang[model], competitor=ang[comp],
                              gated_minus_comp=stats([Jx[S]['grids'][g]['angle_deg'][gk] - Jx[S]['grids'][g]['angle_deg'][comp] for S in xs]),
                              gated_minus_model=stats([Jx[S]['grids'][g]['angle_deg'][gk] - Jx[S]['grids'][g]['angle_deg'][model] for S in xs]))
        r['all_tau_test'] = {t: dict(vs_comp=stats(diff(Jx, xs, g, gate_key(t, comp), comp)), vs_model=stats(diff(Jx, xs, g, gate_key(t, comp), model)),
                                     share=float(np.mean([share(Jx, S, g, t) for S in xs]))) for t in TAUS}
        per = np.array([[np.nan if x is None else x for x in dcontrast(Jx[S]['grids'][g]['scores'], model, comp, 'cwlP')[1]] for S in xs], float)
        r['bins_cwlP'] = dict(mean=np.nanmean(per, 0).tolist(), n=np.isfinite(per).sum(0).tolist())
        L.append(f"| {g} mm | {100 * r['share_model']['mean']:.0f}% | {fm(r['D_cwlP'].get('mean'))} ({r['D_cwlP'].get('n_neg', '–')}/{r['D_cwlP']['n']}, p {fp(r['D_cwlP'].get('p'))}) | "
                 f"{fm(r['D_cwlM'].get('mean'))} ({r['D_cwlM'].get('n_neg', '–')}/{r['D_cwlM']['n']}) | {ang[gk]:.2f} / {ang[model]:.2f} / {ang[comp]:.2f} | "
                 + ', '.join(f"{t}: {fm(r['all_tau_test'][t]['vs_model'].get('mean'))}" for t in TAUS) + ' |')
    L += ['', 'All τ on the test subjects (descriptive; mean Δ against the competitor, subjects better, share sent to the model):', '',
          '| grid | ' + ' | '.join(f'τ = {t}' for t in TAUS) + ' |', '|---|' + '---|' * len(TAUS)]
    for g in GRIDS:
        r = res['grids'][g]
        L.append(f'| {g} mm | ' + ' | '.join(f"{fm(r['all_tau_test'][t]['vs_comp'].get('mean'))} ({r['all_tau_test'][t]['vs_comp'].get('n_neg', '–')}/{r['all_tau_test'][t]['vs_comp']['n']}; {100 * r['all_tau_test'][t]['share']:.0f}%)" for t in TAUS) + ' |')
    L += ['', 'Gain (model − competitor) by CWL-P bin (responsive vertices with < 2% far leak at 2.0 mm), mean (subjects with the bin):', '',
          '| grid | 0–2% | 2–5% | 5–10% | 10–20% | > 20% |', '|---|---|---|---|---|---|']
    for g in GRIDS:
        b = res['grids'][g]['bins_cwlP']
        L.append(f'| {g} mm | ' + ' | '.join(f'{fm(m)} ({n})' if n else '–' for m, n in zip(b['mean'], b['n'])) + ' |')
    # ---- B. registration and the pericalcarine shortfall
    xs = sorted(Jx)
    F = {k: np.array([np.nan if Jx[S]['field'].get(k) is None else Jx[S]['field'][k] for S in xs], float)
         for k in ('bbr_cost', 'p95_brain_hz', 'median_responsive_hz', 'median_calc_responsive_hz')}
    res['registration']['subjects'] = xs; res['registration']['predictors'] = {k: v.tolist() for k, v in F.items()}
    L += ['', '## B. Registration, field and the pericalcarine shortfall (descriptive, test subjects)', '',
          'Predictors: ' + '; '.join(f"{k} median {np.nanmedian(v):.3g} (range {np.nanmin(v):.3g}–{np.nanmax(v):.3g})" for k, v in F.items()) + '.', '',
          '| grid | outcome (model − competitor) | mean | BBR cost: Spearman ρ (p) | field p95 brain: ρ (p) | field at responsive pericalcarine: ρ (p) | '
          'mean, BBR cost ≤ median / > median (Mann–Whitney p) |', '|---|---|---|---|---|---|---|']
    for g in GRIDS:
        model, comp = T['grids'][g]['model'], T['grids'][g]['competitor']; res['registration'][g] = {}
        for oc, lab in (('responsive', 'Δ responsive'), ('calc', 'Δ pericalcarine')):
            y = np.array([np.nan if v is None else v for v in diff(Jx, xs, g, model, comp, oc)], float)
            row = dict(mean=float(np.nanmean(y)), values=y.tolist())
            for k in ('bbr_cost', 'p95_brain_hz', 'median_calc_responsive_hz', 'median_responsive_hz'):
                ok = np.isfinite(y) & np.isfinite(F[k])
                rho, p = spearmanr(F[k][ok], y[ok]); row[k] = dict(rho=float(rho), p=float(p), n=int(ok.sum()))
            ok = np.isfinite(y); med = np.median(F['bbr_cost'][ok]); lo = y[ok & (F['bbr_cost'] <= med)]; hi = y[ok & (F['bbr_cost'] > med)]
            row['split'] = dict(median_bbr=float(med), low=stats(lo.tolist()), high=stats(hi.tolist()), p_mw=float(mannwhitneyu(lo, hi).pvalue))
            res['registration'][g][oc] = row
            L.append(f"| {g} mm | {lab} | {fm(row['mean'])} | {fr(row['bbr_cost']['rho'], row['bbr_cost']['p'])} | {fr(row['p95_brain_hz']['rho'], row['p95_brain_hz']['p'])} | "
                     f"{fr(row['median_calc_responsive_hz']['rho'], row['median_calc_responsive_hz']['p'])} | {fm(row['split']['low'].get('mean'))} (n {row['split']['low']['n']}) / "
                     f"{fm(row['split']['high'].get('mean'))} (n {row['split']['high']['n']}) ({row['split']['p_mw']:.2f}) |")
    # pericalcarine vs the rest: is the pericalcarine set more strongly mixed, less responsive?
    L += ['', 'Pericalcarine set (test subjects, mean over subjects): responsive vertices and their share of the responsive set.', '']
    for g in GRIDS:
        n_calc = [np.mean([Jx[S]['grids'][g]['n'][d]['calc'] for d in ('AB', 'BA')]) for S in xs]
        n_resp = [np.mean([Jx[S]['grids'][g]['n'][d]['responsive'] for d in ('AB', 'BA')]) for S in xs]
        res['registration'][g]['n_calc'] = float(np.mean(n_calc)); res['registration'][g]['share_calc'] = float(np.mean(np.array(n_calc) / np.array(n_resp)))
        L.append(f"- {g} mm: {np.mean(n_calc):.0f} pericalcarine responsive vertices per subject ({100 * res['registration'][g]['share_calc']:.1f}% of the responsive set).")
    json.dump(res, open(OUTP + '.json', 'w'), indent=1, default=float)
    open(OUTP + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
