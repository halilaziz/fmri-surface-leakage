"""Week 18 summary (local), as PLAN_ATLAS.md: contamination from the anatomical atlas (CWL-A) at 3T.
  python3 src/18_atlas_contamination/atlas_summary.py [IN_DIR]     (default results/week18/emul_atlas)
Tests A1/A2: D_CWL-A < 0 at 3.0 / 4.0 mm (35 test subjects; one-sided Wilcoxon, bootstrap 95%, Holm across the two).
Descriptive: D_CWL-A - D_CWL-M and - D_CWL-P, gain by CWL-A bin, Spearman of CWL-A and CWL-P with CWL-M, the V1-V3 variant.
Check: the rerun reproduces the week-17b scores, D_CWL-M and D_CWL-P (results/week17/emul_gate).
Writes results/week18/atlas_test.md / .json.
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('GEO_PROJECT', os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '17_nyu_3t_replication'))
from nyu_summary import load, stats, holm, dcontrast, fm, fp, SPLIT, GRIDS, BN

IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results/week18/emul_atlas')
GATE = os.path.join(ROOT, 'results/week17/emul_gate')
OUTP = os.path.join(ROOT, 'results/week18/atlas_test')
T = json.load(open(os.path.join(ROOT, 'results/week17/nyu_tuning.json')))


def D(J, S, g, var):
    m, c = T['grids'][g]['model'], T['grids'][g]['competitor']
    return dcontrast(J[S]['grids'][g]['scores'], m, c, var)[0]


def main():
    Jx = load(IN, SPLIT['test']); Jg = load(GATE, SPLIT['test']); xs = sorted(Jx)
    res = dict(subjects=xs, grids={}, reproducibility={})
    L = [f'# Week 18: contamination from the anatomical atlas at 3T (PLAN_ATLAS.md) — {len(xs)} test subjects', '',
         'D = gain(highest bin with ≥ 30 vertices) − gain(0–2% bin), gain = residual(model) − residual(competitor); negative: the gain grows',
         'with the dose. One-sided Wilcoxon (less), bootstrap 95% interval of the mean (10,000 resamples, seed 0), Holm across A1 and A2.', '']
    # ---- reproducibility against the week-17b rerun (same settings, same code path)
    for g in GRIDS:
        m, c = T['grids'][g]['model'], T['grids'][g]['competitor']; common = sorted(set(Jx) & set(Jg))
        dd = [abs(Jx[S]['grids'][g]['scores'][k]['responsive'] - Jg[S]['grids'][g]['scores'][k]['responsive']) for S in common for k in (m, c)]
        dm = [abs(D(Jx, S, g, v) - D(Jg, S, g, v)) for S in common for v in ('cwlM', 'cwlP') if D(Jx, S, g, v) is not None and D(Jg, S, g, v) is not None]
        res['reproducibility'][g] = dict(n_subjects=len(common), max_abs_score_diff=float(max(dd)) if dd else None, max_abs_D_diff=float(max(dm)) if dm else None)
    L += ['Reproducibility of the week-17b rerun (model and competitor residuals, D_CWL-M, D_CWL-P): ' + '; '.join(
        f"{g} mm: largest difference {r['max_abs_score_diff']:.1e} (residuals), {r['max_abs_D_diff']:.1e} (D) over {r['n_subjects']} subjects"
        for g, r in res['reproducibility'].items()) + '.', '',
        '## Tests', '', '| grid | test | mean | subjects with D < 0 | bootstrap 95% | p (one-sided) | p (Holm, 2) |', '|---|---|---|---|---|---|---|']
    prim = {}
    for g in GRIDS:
        r = res['grids'][g] = {}
        for var in ('cwlA', 'cwlM', 'cwlP', 'cwlA3'):
            v = [D(Jx, S, g, var) for S in xs]; r[f'D_{var}'] = dict(stats(v), values=v)
        prim[g] = r['D_cwlA'].get('p', 1.0)
    hp = holm(prim)
    for k, g in enumerate(GRIDS):
        st = res['grids'][g]['D_cwlA']; st['p_holm'] = hp.get(g)
        L.append(f"| {g} mm | A{k + 1}: D_CWL-A < 0 | {fm(st.get('mean'))} | {st.get('n_neg', '–')}/{st['n']} | "
                 f"[{fm(st['ci95'][0])}, {fm(st['ci95'][1])}] | {fp(st.get('p'))} | {fp(st['p_holm'])} |")
    # ---- descriptive
    L += ['', '## Descriptive', '', 'Dose contrasts with each response estimate (mean, subjects with D < 0, one-sided p):', '',
          '| grid | CWL-A (atlas, all areas) | CWL-A3 (atlas, V1–V3) | CWL-M (native map) | CWL-P (coarse data) | D_CWL-A − D_CWL-M | D_CWL-A − D_CWL-P |',
          '|---|---|---|---|---|---|---|']
    for g in GRIDS:
        r = res['grids'][g]
        cell = lambda k: f"{fm(r[k].get('mean'))} ({r[k].get('n_neg', '–')}/{r[k]['n']}, p {fp(r[k].get('p'))})"
        for other in ('cwlM', 'cwlP'):
            dv = [(a - b) if a is not None and b is not None else None for a, b in zip(r['D_cwlA']['values'], r[f'D_{other}']['values'])]
            r[f'A_minus_{other}'] = stats(dv, alt='two-sided')
        dcell = lambda k: f"{fm(r[k].get('mean'))} (two-sided p {fp(r[k].get('p'))})"
        L.append(f"| {g} mm | {cell('D_cwlA')} | {cell('D_cwlA3')} | {cell('D_cwlM')} | {cell('D_cwlP')} | {dcell('A_minus_cwlM')} | {dcell('A_minus_cwlP')} |")
    L += ['', 'Gain (model − competitor) by CWL-A bin (responsive vertices with < 2% far leak at 2.0 mm), mean (subjects with the bin):', '',
          '| grid | ' + ' | '.join(['0–2%', '2–5%', '5–10%', '10–20%', '> 20%']) + ' |', '|---|---|---|---|---|---|']
    for g in GRIDS:
        m, c = T['grids'][g]['model'], T['grids'][g]['competitor']
        per = np.array([[np.nan if x is None else x for x in dcontrast(Jx[S]['grids'][g]['scores'], m, c, 'cwlA')[1]] for S in xs], float)
        res['grids'][g]['bins_cwlA'] = dict(mean=np.nanmean(per, 0).tolist(), n=np.isfinite(per).sum(0).tolist())
        L.append(f'| {g} mm | ' + ' | '.join(f'{fm(mu)} ({n})' if n else '–' for mu, n in zip(res['grids'][g]['bins_cwlA']['mean'], res['grids'][g]['bins_cwlA']['n'])) + ' |')
    L += ['', 'How close each measure comes to the fine-data measure: Spearman correlation with CWL-M over the responsive vertices, median',
          'over subjects and directions (and the share of responsive vertices that receive an atlas response):', '',
          '| grid | CWL-A vs CWL-M | CWL-A3 vs CWL-M | CWL-P vs CWL-M | responsive vertices with an atlas response |', '|---|---|---|---|---|']
    for g in GRIDS:
        I = [Jx[S]['grids'][g]['info'].get(d + '_atlas', {}) for S in xs for d in ('AB', 'BA')]
        med = lambda k: float(np.nanmedian([np.nan if x.get(k) is None else x[k] for x in I]))
        res['grids'][g]['spearman'] = {k: med(k) for k in ('rho_A_M', 'rho_A3_M', 'rho_P_M', 'share_resp_with_atlas_response')}
        sp = res['grids'][g]['spearman']
        L.append(f"| {g} mm | {sp['rho_A_M']:.2f} | {sp['rho_A3_M']:.2f} | {sp['rho_P_M']:.2f} | {100 * sp['share_resp_with_atlas_response']:.0f}% |")
    json.dump(res, open(OUTP + '.json', 'w'), indent=1, default=float)
    open(OUTP + '.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
