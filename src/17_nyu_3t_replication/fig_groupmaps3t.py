"""Week 17b C figure (local): group far-leak maps of the NYU 3T 2.0 mm acquisition and its 3.0 / 4.0 mm emulations on the
fsaverage inflated surface, in the style of the HCP group maps (week 7, Fig. 3): lateral and medial views, mean far-leak fraction,
one colour bar (0-0.3), medial wall and uncovered vertices grey.
Usage: python3 src/17_nyu_3t_replication/fig_groupmaps3t.py [DIR]   (default results/week17/groupmaps; needs DIR/_draw from nyu_groupmaps.py)
"""
import sys, os, json
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

D = sys.argv[1] if len(sys.argv) > 1 else 'results/week17/groupmaps'
GRIDS = ('2.0', '3.0', '4.0')


def main():
    summ = json.load(open(f'{D}/nyu_groupmaps.json')); nsub = summ['n']
    fig, ax = plt.subplots(len(GRIDS), 4, figsize=(15, 3.0 * len(GRIDS) + 0.8))
    for gi, g in enumerate(GRIDS):
        for hi, h in enumerate('LR'):
            S = np.load(f'{D}/_draw/fsaverage_{h}_inflated.npz'); V, F = S['V'].astype(float), S['F'].astype(np.int64)
            val = np.asarray(nib.load(f'{D}/tpl-fsaverage_hemi-{h}_den-164k_desc-farleak3T{g.replace(".", "p")}mm-mean.func.gii').darrays[0].data, float)
            fv = val[F]; fc = np.ma.masked_where(~np.isfinite(fv).all(1), np.where(np.isfinite(fv), fv, 0).mean(1))
            cm = plt.get_cmap('magma_r').copy(); cm.set_bad('0.8')
            fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            for vi, (side, sgn) in enumerate((('lateral', -1 if h == 'L' else 1), ('medial', 1 if h == 'L' else -1))):
                a = ax[gi, 2 * hi + vi]
                vis = fn[:, 0] * sgn > 0
                order = np.argsort(sgn * V[F, 0].mean(1)); order = order[vis[order]]
                xs = V[:, 1] * sgn
                tpc = a.tripcolor(Triangulation(xs, V[:, 2], F[order]), facecolors=fc[order], cmap=cm, vmin=0, vmax=0.3)
                a.set_aspect('equal'); a.axis('off')
                a.set_title(f'{h} {side}, {g} mm' + (' (acquired)' if g == '2.0' else ' (emulated)'), fontsize=9)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.02, hspace=0.12)
    cax = fig.add_axes([0.35, 0.05, 0.30, 0.018])
    fig.colorbar(tpc, cax=cax, orientation='horizontal', label='far-leak fraction of standard mapping (share of the mapped signal from > 15 mm or the other hemisphere)')
    fig.suptitle(f'Operator-predicted leakage of standard mapping, NYU 3T 2.0 mm acquisition and emulations (mean over {nsub} subjects, fsaverage)', fontsize=10)
    out = os.path.join(D, 'figW17-2_group_leakage_3T.png'); plt.savefig(out, dpi=140); plt.close(); print(out)


if __name__ == '__main__':
    main()
