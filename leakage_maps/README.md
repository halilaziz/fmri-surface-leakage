# Group leakage maps (HCP 7T on fs_LR 32k; NYU 3T on fsaverage)

Operator-predicted far leak of standard ribbon-constrained surface mapping. This is the share of a vertex's mapped value that
comes from cortex more than 15 mm away along the surface or in the other hemisphere, for a unit response field. Settings: HCP 7T
geometry and point-spread function, 1.6 mm native voxels, and 2.4 / 3.2 mm voxels emulated in k-space. Values are the mean over 19
HCP subjects, after resampling each subject's native-mesh map to fs_LR 32k with its MSMAll sphere (area-corrected barycentric).
The medial wall (outside most subjects' atlas ROI) is 0.

Files:
- `group_leakage.fsLR32k.L.func.gii`, `group_leakage.fsLR32k.R.func.gii`, nine maps each:
  - `leak_mean_{1.6,2.4,3.2}`: mean far-leak fraction;
  - `leak_sd_{grid}`: s.d. over subjects;
  - `frac_subjects_leak_gt10_{grid}`: fraction of subjects with far leak > 10%.
- `fsLR32k.{L,R}.veryinflated.surf.gii`: display surfaces (generated from the fs_LR 32k midthickness with
  `wb_command -surface-generate-inflated`).
- `group_leakage_summary.json`: cortex fractions above 5 / 10 / 20% per grid, and area-weighted means per Desikan region.
- `figW7-1_group_leakage.png`: the figure in the paper.

Load in Python:

```python
import nibabel as nib
g = nib.load('group_leakage.fsLR32k.L.func.gii')
maps = {d.meta['Name']: d.data for d in g.darrays}      # e.g. maps['leak_mean_2.4']
```

Or open in Connectome Workbench (`wb_view`) together with the surfaces.

These maps describe HCP 7T geometry and PSF. For another acquisition, rebuild the operator with that acquisition's parameters
(week-2 code).

## 3T acquisition (NYU Retinotopy Dataset), fsaverage 164k — `3T_nyu/`

Operator-predicted far leak of standard mapping for the NYU 3T protocol (2.0 mm isotropic, multiband EPI; Himmelberg et al.,
2021) and for its emulated 3.0 and 4.0 mm versions (k-space crop and taper, as for HCP). Per subject the far-leak map of the
analysis vertices was carried from the subject's FreeSurfer registration sphere (`?h.sphere.reg`) to fsaverage's by the three
nearest vertices (inverse-distance weights); vertices covered in fewer than half of the subjects (medial wall) are NaN.

Files (`hemi` L / R; grid `2p0`, `3p0`, `4p0` mm):
- `tpl-fsaverage_hemi-{L,R}_den-164k_desc-farleak3T{grid}mm-mean.func.gii`: mean far-leak fraction;
- `tpl-fsaverage_hemi-{L,R}_den-164k_desc-farleak3T{grid}mm-share-gt5pct.func.gii`: share of subjects above 5%;
- `nyu_groupmaps.json` / `.md`: cortex fractions above 5 / 10% of the group maps and, per Desikan region, the area-weighted mean
  far leak and the share above 10% (computed in each subject's space, then averaged);
- `figW17-2_group_leakage_3T.png`: Figure S2 of the paper.

Display them on FreeSurfer's fsaverage surfaces (`$FREESURFER_HOME/subjects/fsaverage/surf/?h.inflated`, not included here).

