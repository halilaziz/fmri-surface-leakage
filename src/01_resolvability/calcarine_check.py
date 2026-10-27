"""Bank proximity and predicted ribbon-mapping leakage inside the pericalcarine label vs whole cortex (week-1 maps)."""
import os, sys, glob, json
import numpy as np, nibabel as nib
MAPS = sys.argv[1]
HCP = os.environ.get('GEO_ROOT', '/path/to/geo_root') + '/data/hcp_extra'
rows = []
for sd in sorted(glob.glob(os.path.join(MAPS, '[0-9]' * 6))):
    S = os.path.basename(sd)
    for h in 'LR':
        m = {da.meta['Name']: np.asarray(da.data) for da in nib.load(f'{sd}/{S}.{h}.week1_maps.func.gii').darrays}
        lab = nib.load(f'{HCP}/{S}/MNINonLinear/Native/{S}.{h}.aparc.native.label.gii')
        key = [k for k, v in lab.labeltable.get_labels_as_dict().items() if 'pericalcarine' in v.lower()][0]
        g = nib.load(f'{sd}/{S}.{h}.midthickness.acpc.surf.gii')
        V = g.agg_data('NIFTI_INTENT_POINTSET'); F = g.agg_data('NIFTI_INTENT_TRIANGLE')
        fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        a = np.zeros(len(V))
        for k in range(3):
            np.add.at(a, F[:, k], fa / 3)
        cortex = m['thickness'] > 0
        calc = cortex & (lab.darrays[0].data == key)
        def st(mask):
            w = a[mask] / a[mask].sum(); d = m['d_opp'][mask]
            o = np.argsort(d); med = d[o][np.searchsorted(np.cumsum(w[o]), 0.5)]
            return dict(dopp_median=float(med), frac_lt3=float(w[d < 3].sum()), frac_lt4=float(w[d < 4].sum()),
                        leak3T=float((w * m['leak_naive_3T'][mask]).sum()), leak7T=float((w * m['leak_naive_7T'][mask]).sum()),
                        thickness=float((w * m['thickness'][mask]).sum()))
        rows.append(dict(subject=S, hemi=h, calc=st(calc), cortex=st(cortex), calc_area_cm2=float(a[calc].sum() / 100)))
out = {}
for reg in ('calc', 'cortex'):
    out[reg] = {k: (float(np.mean([r[reg][k] for r in rows])), float(np.std([r[reg][k] for r in rows]))) for k in rows[0][reg]}
out['calc_area_cm2'] = (float(np.mean([r['calc_area_cm2'] for r in rows])), float(np.std([r['calc_area_cm2'] for r in rows])))
json.dump(dict(per_hemisphere=rows, group=out), open(os.path.join(MAPS, '_summary', 'week1_calcarine_check.json'), 'w'), indent=1)
for reg, lab in (('calc', 'pericalcarine'), ('cortex', 'whole cortex')):
    g = out[reg]
    print(f"{lab:14s} median d_opp {g['dopp_median'][0]:.2f}±{g['dopp_median'][1]:.2f} mm | banks <3 mm {100*g['frac_lt3'][0]:.1f}% | <4 mm {100*g['frac_lt4'][0]:.1f}% "
          f"| leakage 3T {100*g['leak3T'][0]:.2f}% 7T {100*g['leak7T'][0]:.2f}% | thickness {g['thickness'][0]:.2f} mm")
print(f"pericalcarine area per hemisphere: {out['calc_area_cm2'][0]:.1f} ± {out['calc_area_cm2'][1]:.1f} cm2")
