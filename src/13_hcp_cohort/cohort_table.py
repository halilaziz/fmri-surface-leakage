"""Condensed markdown table of the cohort tests (primary set and all subjects) from cohort_stats.py's JSON. Used by the week-13
report and by manuscript/build_v1.py (Section 4.9), so the numbers are formatted once from the result file and never retyped.
Usage: python3 src/13_hcp_cohort/cohort_table.py results/week13/cohort_final.json
"""
import json, sys

ROWS = [  # (hypothesis, role, label in the JSON, name in the table, grid)
 ('H1', 'primary', 'D_CWL-M < 0, 2.4 mm', 'D, contamination (native-map estimate)', '2.4'),
 ('H1', 'primary', 'D_CWL-M < 0, 3.2 mm', 'D, contamination (native-map estimate)', '3.2'),
 ('H1', 'secondary', 'D_CWL-P < 0, 2.4 mm', 'D, contamination (coarse data only)', '2.4'),
 ('H1', 'secondary', 'D_CWL-P < 0, 3.2 mm', 'D, contamination (coarse data only)', '3.2'),
 ('H1', 'secondary', 'D_far < 0, 2.4 mm', 'D, plain far leak', '2.4'),
 ('H1', 'secondary', 'D_far < 0, 3.2 mm', 'D, plain far leak', '3.2'),
 ('H1', 'secondary', 'gain, all coherent < 0, 2.4 mm', 'overall gain, all coherent vertices', '2.4'),
 ('H1', 'secondary', 'gain, all coherent < 0, 3.2 mm', 'overall gain, all coherent vertices', '3.2'),
 ('H2', 'primary (H2a)', 'D_far(wedge) − D_far(ring) < 0, 3.2 mm', 'D_far(wedges) − D_far(rings)', '3.2'),
 ('H2', 'interval', 'D_far(wedge) − D_far(ring), 2.4 mm', 'D_far(wedges) − D_far(rings)', '2.4'),
 ('H2', 'primary (H2b)', 'D_CWL-M(ring) < 0, 2.4 mm', 'D, contamination (rings)', '2.4'),
 ('H2', 'primary (H2b)', 'D_CWL-M(ring) < 0, 3.2 mm', 'D, contamination (rings)', '3.2'),
 ('H2', 'secondary (H2c)', 'D_CWL-M(ring) − D_far(ring) < 0, 3.2 mm', 'D_CWL(rings) − D_far(rings)', '3.2'),
 ('H2', 'descriptive', 'D_far(ring), 2.4 mm', 'D, plain far leak (rings)', '2.4'),
 ('H2', 'descriptive', 'D_far(ring), 3.2 mm', 'D, plain far leak (rings)', '3.2'),
 ('H3', 'primary (H3a)', 'Δ_clean(half) − Δ_clean(full) > 0, 2.4 mm', 'Δ_clean(half) − Δ_clean(full), > 0 predicted', '2.4'),
 ('H3', 'primary (H3a)', 'Δ_clean(half) − Δ_clean(full) > 0, 3.2 mm', 'Δ_clean(half) − Δ_clean(full), > 0 predicted', '3.2'),
 ('H3', 'interval (H3b)', 'D_far(half) − D_far(full), 2.4 mm (interval should include 0 or be positive)', 'D_far(half) − D_far(full)', '2.4'),
 ('H3', 'interval (H3b)', 'D_far(half) − D_far(full), 3.2 mm (interval should include 0 or be positive)', 'D_far(half) − D_far(full)', '3.2'),
 ('H3', 'secondary (H3c)', 'Δ_clean(half) < 0, 2.4 mm', 'Δ_clean(half)', '2.4'),
 ('H3', 'secondary (H3c)', 'Δ_clean(half) < 0, 3.2 mm', 'Δ_clean(half)', '3.2'),
 ('H4', 'primary', 'PSF 15% too wide: gain, all coherent < 0, 2.4 mm', 'overall gain, PSF 15% too wide', '2.4'),
 ('H4', 'secondary', 'PSF 15% too wide: gain, all coherent < 0, 3.2 mm', 'overall gain, PSF 15% too wide', '3.2'),
 ('H4', 'secondary', 'PSF 15% too narrow: gain, all coherent < 0, 2.4 mm', 'overall gain, PSF 15% too narrow', '2.4'),
 ('H4', 'secondary', 'PSF 15% too narrow: gain, all coherent < 0, 3.2 mm', 'overall gain, PSF 15% too narrow', '3.2'),
]


def fnum(x, d=4):
    return f'{x:+.{d}f}'.replace('-', '−')


def fci(ci):
    return f'[{fnum(ci[0])}, {fnum(ci[1])}]'


def fp(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def table(path, holm=False):
    J = json.load(open(path)); R = {r['label']: r for r in J['rows']}
    L = ['| hyp. | test | grid | role | n | mean | in direction | bootstrap 95% | p' + (' | p (Holm)' if holm else '')
         + ' | all subjects: n, mean [family bootstrap 95%] |', '|---' * (10 + holm) + '|']
    for h, role, lab, name, g in ROWS:
        r = R[lab]; p = r['primary']; a = r['all']
        tested = not role.startswith(('interval', 'descriptive'))
        cells = [h, name, f'{g} mm', role, str(p['n']), fnum(p['mean']), f"{p['n_dir']}/{p['n']}", fci(p['ci95']), fp(p['p']) if tested else '–']
        if holm:
            cells.append(fp(p['p_holm']) if p.get('p_holm') is not None else '–')
        cells.append(f"{a['n']}, {fnum(a['mean'])} {fci(r['all_family_boot'])}")
        L.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(L)


if __name__ == '__main__':
    print(table(sys.argv[1], holm='--holm' in sys.argv))
