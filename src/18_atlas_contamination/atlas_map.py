"""Week 18 (PLAN_ATLAS.md): the anatomical retinotopy template of Benson et al. (2014; v4.0 files from the neuropythy 0.13.0
source archive, on fsaverage) carried to each NYU subject's FreeSurfer mesh. For every subject vertex, the three nearest fsaverage
vertices on the unit-normalised registration spheres (?h.sphere.reg), inverse-distance weights: polar angle on the unit circle
(as a visual-field angle phi, left hemisphere 90 - angle, right hemisphere 90 + angle, degrees), eccentricity linearly, area
label from the nearest vertex. No neuropythy code is run.
Writes GEO_CONST/results/nyu/atlas/SUBJECT.benson14.npz: phi, eccen, varea for the left then the right hemisphere's vertices
(the order of the concatenated meshes that nyu_emul.py indexes), and nL.
Usage (fMRIPrep container on hebb): python3 atlas_map.py SUBJECT [SUBJECT ...]
"""
import os, sys
import numpy as np, nibabel as nib
from scipy.spatial import cKDTree

G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
FSD = f'{G}/data/ds003787/derivatives/freesurfer'
TPL = f'{G}/data/atlases/benson14_v4_0/surf'
OUTD = f'{G}/results/nyu/atlas'
FSAVG = os.environ.get('FSAVG', '/opt/freesurfer/subjects/fsaverage')


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    os.makedirs(OUTD, exist_ok=True)
    tpl = {}
    for h in ('lh', 'rh'):
        load = lambda p: np.asarray(nib.load(f'{TPL}/{h}.benson14_{p}.v4_0.mgz').dataobj, np.float64).ravel()
        ang, ecc, va = load('angle'), load('eccen'), load('varea')
        phi = np.radians(90.0 - ang if h == 'lh' else 90.0 + ang)
        sph = unit(nib.freesurfer.read_geometry(f'{FSAVG}/surf/{h}.sphere.reg')[0])
        assert len(sph) == len(ang) == 163842
        tpl[h] = dict(tree=cKDTree(sph), c=np.cos(phi), s=np.sin(phi), ecc=ecc, va=va)
    for S in sys.argv[1:]:
        res = {}
        for h in ('lh', 'rh'):
            sph = unit(nib.freesurfer.read_geometry(f'{FSD}/{S}/surf/{h}.sphere.reg')[0])
            d, j = tpl[h]['tree'].query(sph, k=3)
            w = 1.0 / np.maximum(d, 1e-9); w /= w.sum(1, keepdims=True)
            T = tpl[h]
            c, s = (w * T['c'][j]).sum(1), (w * T['s'][j]).sum(1)
            res[h] = dict(phi=np.degrees(np.arctan2(s, c)), ecc=(w * T['ecc'][j]).sum(1), va=T['va'][j[:, 0]].astype(np.int16))
        nL = len(res['lh']['phi'])
        np.savez_compressed(f'{OUTD}/{S}.benson14.npz', phi=np.concatenate([res[h]['phi'] for h in ('lh', 'rh')]).astype(np.float32),
                            eccen=np.concatenate([res[h]['ecc'] for h in ('lh', 'rh')]).astype(np.float32),
                            varea=np.concatenate([res[h]['va'] for h in ('lh', 'rh')]), nL=nL)
        lab = np.concatenate([res[h]['va'] for h in ('lh', 'rh')]); ecc = np.concatenate([res[h]['ecc'] for h in ('lh', 'rh')])
        print(S, 'vertices', len(lab), 'labelled', int((lab > 0).sum()), 'labelled within 12.4 deg', int(((lab > 0) & (ecc <= 12.4)).sum()),
              'V1-V3 within 12.4 deg', int(((lab >= 1) & (lab <= 3) & (ecc <= 12.4)).sum()), flush=True)


if __name__ == '__main__':
    main()
