# Week 16 — addendum to PLAN_BEFG.md: F2, the correlated-noise likelihood with its own prior settings (written before any F2 fit)

**Why, stated openly.** F as planned fits the correlated-noise likelihood at the prior settings chosen for the white-noise
likelihood (12 mm / 0.125 at 2.4 mm, 24 mm / 0.25 at 3.2 mm). In the first three development subjects to finish, the correlated
fit was worse than the white fit in every stratum (all coherent vertices +0.054 at 2.4 mm and +0.025 at 3.2 mm, 0 of 3 better),
and the estimated noise spectrum is dominated by large spatial scales (480–630 times the white level in the lowest radial bin,
0.03–0.04 in the highest). Changing the likelihood changes the balance between data and prior, so a comparison at the white
likelihood's settings confounds the likelihood with its tuning — the confound that E removes for H3. F2 removes it for F. It is
added after seeing those three subjects; F itself is reported as planned.

## Design

- **Fits.** The week-16 correlated-noise fit (`src/week16/hcp_corr.py`, unchanged: spectrum from the eight neighbouring noise bins
  of both runs, floor 0.1 of white, all valid voxels as rows) on a grid of prior settings, range r ∈ {6, 12, 24, 48} mm ×
  amplitude scale s ∈ {0.03125, 0.0625, 0.125, 0.25, 0.5, 1}, both grids (2.4, 3.2 mm), full wedge runs, for the 5 tuning
  subjects and the 15 development subjects. Scored with `src/week16/variant_eval.py` together with the subject's white-noise
  fits from week 3 (tuning subjects: the week-3 tuning grid; development subjects: the chosen settings and the competitors).
- **Selection (tuning subjects only).** Per grid, the correlated-likelihood setting with the lowest mean cross-run residual over
  all coherent vertices across the 5 tuning subjects — the week-3 rule. The white likelihood keeps its week-3 choice.
- **Comparison (15 development subjects).** corr(re-tuned) − white(chosen), per subject, over all coherent vertices (primary
  readout), where leak > 10% (clean at 1.6 mm), in the pericalcarine and in the far-leak bins, and for D_far; mean, subjects in
  favour, bootstrap 95% interval, two-sided Wilcoxon (descriptive, n = 15). Reported alongside: corr at the white settings (F,
  reproduced inside F2 as a check), and the best correlated setting chosen on the 15 development subjects themselves (an
  optimistic bound, not a test).
- **Rule.** If corr(re-tuned) does not have a lower mean residual than white(chosen) over all coherent vertices, the manuscript
  states that the white likelihood did not cost accuracy in HCP emulation and that the correlated likelihood estimated from
  neighbouring frequency bins did not help, with or without re-tuning. If it does, the manuscript reports the improvement, its
  size and where it occurs, and the Limitations entry on the noise model is revised accordingly. Either way D_far under both
  likelihoods is reported.

Hash in `PLAN_F2_ADDENDUM.sha1`, recorded before the first F2 job is submitted.
