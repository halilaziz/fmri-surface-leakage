# Week 15, analysis A — far-part ablation of the inversion (written before any fit; reviewer-requested)

Both reviewer passes asked for the direct test of the paper's mechanism claim: invert with the far part of the forward operator
removed and see whether the dose-dependent gain disappears while the gain where no far mixing exists stays.

## Fixed before the fits

- **Subjects, data, settings.** The 15 development subjects (the week-3 test set), wedge runs (full), emulated 2.4 and 3.2 mm;
  prior 12 mm / 0.125 at 2.4 mm and 24 mm / 0.25 at 3.2 mm; competitor upsampling + 1.6 mm mapping + 8 mm smoothing; the
  cross-run residual against the other run's 1.6 mm map, as in week 3. Nothing is re-tuned.
- **Operators.** Full K as in week 3. *Near-only* K_near: for every voxel, the entries to unknowns that are far from the voxel's
  reference unknown — far = the other hemisphere or more than 15 mm along the sheet on the area-scaled sphere, the definition of
  the leakage maps; reference = the unknown with the largest |K| entry of that voxel — are set to zero. *Renormalised near*
  K_nearn: K_near with each voxel row rescaled to the signed row sum of K (the "far cortex responds like the near cortex" null).
  Everything else is identical: noise weights, prior, PCG tolerance 1e-4, rendering through the full 1.6 mm operator.
- **Quantities.** Per subject and grid, for the full, near-only and renormalised-near models and the competitor: the cross-run
  residual in the far-leak bins of the dose-response (coherent vertices clean at 1.6 mm: 0–2, 2–5, 5–10, 10–20, > 20%), over all
  coherent vertices, where leak > 10% and in the pericalcarine; Δ = model − competitor per bin; D_far = Δ(highest bin with ≥ 30
  vertices) − Δ(0–2%); Δ_clean = Δ(0–2%). Also the share of the operator's mass removed per voxel (descriptive).
- **Predictions and tests** (n = 15; one-sided Wilcoxon where a direction is stated; bootstrap 95% of the mean, 10,000
  resamples, seed 0):
  - P1: the dose-dependent gain shrinks without the far part: D_far(near) − D_far(full) > 0 at both grids.
  - P2: the gain where no far mixing exists is unchanged: Δ_clean(near) − Δ_clean(full), 95% interval; |mean| < 0.003 expected.
  - P3: the near-only model is worse overall: gain(near) − gain(full) > 0 over all coherent vertices, at both grids.
  - P4: P1–P3 repeated for the renormalised variant.
- **Interpretation, stated in advance.** P1 and P2 both holding means the dose-dependent gain is far unmixing and the
  leak-independent gain is near-field (deconvolution within 15 mm and exact ribbon sampling). P1 failing (D_far unchanged
  without the far part) means the dose-response is not produced by the far part of the operator, and the mechanism claim is
  withdrawn. P2 failing (the clean-bin gain also shrinks) means the far part matters even where the far leak is < 2%.
- **Status.** Exploratory, reviewer-requested, on the development set; reported with its n in the main text and in full in the
  Supplementary material. Hash recorded in `ABLATION_PLAN.sha1` before the first fit; the job driver refuses to submit fits
  without it.
