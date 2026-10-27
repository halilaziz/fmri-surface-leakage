# Week 16 — the remaining reviewer-requested analyses B, E, F, G (written before any fit)

## B. PSF-scale scan (5 tuning subjects)

- **Fits.** Chosen settings (12 mm / 0.125 at 2.4 mm, 24 mm / 0.25 at 3.2 mm), wedge runs, with the inversion's coarse-grid
  kernels stretched by α = 0.7, 0.85, 1.15, 1.3 on all axes, and by 0.85 and 1.15 on the phase-encoding axis only (y) and on
  the slice axis only (z); α = 1 from the week-3 tuning fits. Data and rendering unchanged (as in the week-3 ±15% test).
- **Readout.** Cross-run residual over all coherent vertices, where leak > 10% (clean at 1.6 mm) and in the pericalcarine,
  per α and axis set; mean over the 5 subjects and the subject count favouring each α over α = 1.
- **Expectation stated in advance.** If the modelled native PSF is right, the residual is lowest at α = 1 on every axis set.
  The week-3 and cohort results (−15% fits slightly better) predict a minimum below 1; the axis sets say which axis drives it.
  Descriptive (n = 5); no test. The PSF-component table (`psf_components.md`) gives each component's FWHM contribution and
  the provenance of each parameter.

## E. H3 with settings re-tuned for half-run sources

- **E1 (5 tuning subjects).** Fits from each half run (4 cycles) at 16 prior settings per grid (range 6, 12, 24, 48 mm ×
  amplitude scale 0.0625, 0.125, 0.25, 0.5), competitors at smoothing FWHM 2–16 mm; the week-3 selection rule (lowest mean
  residual over the 5 subjects, all coherent vertices, halves averaged) picks the half-run settings of the model, of standard +
  smoothing and of upsampling + smoothing.
- **E2 (15 development subjects).** Δ_clean(half) and D_far(half) recomputed with the re-tuned settings (model refitted at the
  re-tuned prior if it differs from the full-run one; competitors from the saved arrays), against Δ_clean(full) and D_far(full)
  at the full-run settings. H3a′: Δ_clean(half, re-tuned) − Δ_clean(full) > 0 (one-sided Wilcoxon, n = 15); H3b′ as an interval.
- **Rule.** If H3a′ holds, H3's conclusion ("not denoising") does not depend on the fixed-settings confound. If it reverses
  (the gain where no far mixing exists grows with noisier sources once both methods are tuned for them), the denoising
  explanation revives and the manuscript's statement is withdrawn.

## F. Correlated-noise likelihood on the HCP development set (15 subjects)

- **Fits.** The week-6 likelihood (stationary noise correlation, spectrum from the data's own noise, applied with FFTs; `w6lib`)
  at the chosen settings, both grids; the spectrum from the complex Fourier components at the eight noise bins neighbouring the
  stimulus frequency, each scaled to unit variance per voxel, on the coarse grid of each run.
- **Readout.** Residual over all coherent vertices, where leak > 10%, in the far-leak bins; D_far; against the white-noise
  model and the competitor. Paired differences corr − white (two-sided, n = 15; descriptive).
- **Expectation stated in advance.** From week 6 (StudyForrest): small changes, if any, concentrated at coarse grids. The
  question is whether the misspecified white likelihood costs accuracy or changes the dose-response; either way the result is
  reported.

## G. The "far" criterion: true geodesics and the 15 mm threshold

- **G1 (20 development subjects, operator only).** The leakage maps recomputed with the sphere proxy at thresholds 10, 15 and
  20 mm, and with "far = other hemisphere only"; mean far leak, cortex above 5% and 10%, pericalcarine, at 1.6 / 2.4 / 3.2 mm.
- **G2 (5 tuning subjects).** The far criterion recomputed with true geodesic distances on the midthickness mesh (Dijkstra on
  the triangle graph, limit 25 mm; pairs beyond the limit are far) for the same operator rows; per-vertex agreement of the
  geodesic and sphere-proxy leak fractions (correlation, mean difference, share of vertices whose > 5% / > 10% label changes),
  the share of operator mass whose far/near label flips, and the resulting group-figure differences.
- Descriptive.

Hashes in `PLAN_BEFG.sha1`, recorded before the first job of B, E and F was submitted (G is operator-only and descriptive).
