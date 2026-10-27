# Week 18 — contamination from an anatomical atlas at 3T (written before anything is computed)

User's request (2026-10-13): "run the atlas-based contamination test". In the NYU Retinotopy data, contamination computed with
the source half's native-resolution map as the response estimate (CWL-M) predicted the model's gain at both coarsenings (N3,
PLAN_NYU.md), while contamination computed from the coarse data alone (CWL-P) did not (PLAN_NYU_GATE.md, descriptive). The
question here: does contamination computed from an anatomical retinotopy atlas, which uses no functional data of the subject,
predict where the model gains at 3T? If it does, a study could map where correction should pay before acquiring data.

## Measure

- **Atlas.** The anatomical retinotopy template of Benson et al. (2014), version 4.0 as distributed with neuropythy 0.13.0
  (PyPI source archive; files `?h.benson14_{angle,eccen,varea}.v4_0.mgz` on fsaverage, 163,842 vertices per hemisphere; polar
  angle 0–180° from the upper to the lower vertical meridian in both hemispheres; visual areas 1–12). Only the template files
  are used; no neuropythy code is run.
- **Transfer to the subject.** For every vertex of the subject's FreeSurfer mesh, the three nearest fsaverage vertices on the
  registration spheres (`?h.sphere.reg`, unit-normalised), inverse-distance weights: polar angle interpolated on the unit circle,
  eccentricity linearly, the area label from the nearest vertex.
- **Response estimate.** x = exp(iφ) where the vertex has an area label (1–12) and eccentricity ≤ 12.4° (the NYU stimulus radius,
  Himmelberg et al., 2021), x = 0 elsewhere; φ is the polar angle in one visual-field convention for both hemispheres (left
  hemisphere φ = 90° − angle, right hemisphere φ = 90° + angle). CWL depends only on differences of x, so the choice of phase
  origin and direction does not matter.
- **CWL-A.** The contamination-weighted leak of standard mapping at the coarse grid with this x, computed exactly as CWL-M and
  CWL-P (same far criterion, normalisation by the RMS of x over the responsive vertices, bins 0–2, 2–5, 5–10, 10–20, > 20%,
  responsive vertices with < 2% far leak at 2.0 mm, ≥ 30 vertices per bin). D_CWL-A = gain in the highest bin with ≥ 30
  vertices − gain in the 0–2% bin, gain = residual(model) − residual(competitor).

## Data, settings, subjects

As PLAN_NYU.md and PLAN_NYU_GATE.md: the 35 included test subjects; model 12 mm / 2 at 3.0 mm and 6 mm / 1 at 4.0 mm;
competitor upsampling + 2 mm smoothing; responsive vertices R² ≥ 0.2. `nyu_emul.py` reruns every test subject at these settings
with the atlas response added to the contamination pass; the published scores, D_CWL-M and D_CWL-P must be reproduced exactly
(check).

## Tests (35 test subjects)

- **A1:** D_CWL-A < 0 at 3.0 mm. **A2:** D_CWL-A < 0 at 4.0 mm.
- One-sided Wilcoxon signed-rank test, bootstrap 95% interval of the mean (10,000 resamples, seed 0), Holm across the two.
- Interpretation fixed now: if A1 and A2 hold, an anatomy-only map predicts where correction pays at 3T; if both fail, the 3T
  dose-response needs a response map finer than the coarse data; one of two is reported as such, grid by grid.

## Descriptive

- D_CWL-A − D_CWL-M and D_CWL-A − D_CWL-P per subject (two-sided Wilcoxon).
- Gain by CWL-A bin (mean over subjects with the bin).
- Per subject, Spearman correlation over the responsive vertices of CWL-A with CWL-M, and of CWL-P with CWL-M (median over
  subjects): how close each coarse-free or coarse-only measure comes to the fine-data measure.
- Variant: the atlas restricted to V1–V3 (areas 1–3), D_CWL-A3.

## Not part of this plan

HCP: the per-subject HCP surfaces on TRUBA were cleared after the cohort run; an HCP test would follow under its own plan, and
only if A1 and A2 hold. No gated estimator is run.

## Compute

hebb, inside `/analysis/halilaziz/halil/GEO_CONST`: template files extracted to `data/atlases/benson14_v4_0/surf/`; per-subject
atlas maps by `src/week18/atlas_map.py` (fMRIPrep container, FreeSurfer's fsaverage) into `results/nyu/atlas/`; scores by
`nyu_emul.py --atlas` into `results/nyu/emul_atlas/`; summary `src/week18/atlas_summary.py` locally; results copied to
`results/week18/`.

Hash in `PLAN_ATLAS.sha1`, recorded before the first run.
