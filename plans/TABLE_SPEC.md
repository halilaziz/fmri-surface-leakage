# Week 11: one set of statistics across datasets (written and hashed before computing)

**Purpose.** Restate every main model-versus-competitor comparison of the study with the same statistics, so the datasets and
conditions can be compared at a glance. The table is descriptive. It summarises analyses that were already run; only the
high-mixing sets of the ds002702 within-session rows (A4–A6) are computed for the first time.

## Quantity

Per subject, d = residual(model) − residual(best competitor); d < 0 when the model is better. The best competitor is the better
of the competitor families for that subject, under the same condition as the model:
- HCP: standard + surface smoothing, or upsampling + 1.6 mm mapping + smoothing;
- StudyForrest within session: smoothing, or upsampling (± smoothing);
- ds002702 within session: the competitor chosen leave-one-subject-out among upsampling ± 2 / 4 mm and coarse map + 4 / 8 mm;
- cross-session rows: the competitor of that analysis.

Where a subject contributes several cells (StudyForrest sessions × coarsening factors) or directions, d is first averaged within
the subject with equal weights. The subject is the unit of analysis.

## Statistics (every row, as in the pre-declared tests)

n subjects; mean and median of d; subjects with d < 0; bootstrap 95% interval of the mean (10,000 resamples, a fresh generator
with seed 0 for every row); one-sided Wilcoxon signed-rank p (alternative: d < 0; scipy, exact for small n). No correction for
multiplicity. With n = 5 the smallest possible p is 0.031; with n = 2 it is 0.25.

## Vertex sets (two per row)

- **all:** coherent vertices (HCP) or visual vertices (StudyForrest, ds002702);
- **high mixing:** far leak > 10% at the coarse grid and < 2% at the native grid (HCP, ds002702); > 10% of the coarse map from the
  other hemisphere and < 2% at the native grid (StudyForrest within session); source far leak > 10% and < 2% at the target grid
  (ds002702 cross-session). Not available for StudyForrest cross-session.

Settings follow each analysis' own rule; no row uses target data to choose settings.

## Rows

**A. Emulation within one acquisition**
- A1, A2. HCP 7T, emulated 2.4 / 3.2 mm, 15 test subjects; settings fixed on 5 tuning subjects.
  Source `results/week3/week3_test_scores.json`: model − fmin(smoothing, upsample_smoothing); sets coherent / clean16_leak10.
- A3. StudyForrest 7T within session, 1.4 / 2.0 mm sessions coarsened ×1.5 / ×2, 7 subjects; settings leave-one-subject-out;
  correlated-noise model (as in the manuscript's Section 4.5). Source `results/week6/w6_results.json`: corr_loso −
  min(smoothing_loso, upsample_loso) per cell; sets visual / visual_xhemi.
- A4–A6. ds002702 7T within session, 0.8 mm coarsened to 1.05 / 1.6 / 2.4 mm, 5 subjects; settings leave-one-subject-out.
  All: `results/week7/ds002702/summary/kay_results.json` (model − comp, visual). High mixing: computed from
  `results/week7/ds002702/emul/sub-*/*.kay_emul.npz` with the procedure of `src/week7/summarize_kay.py` (same scores, same model
  and competitor candidates, leave-one-subject-out choice on the set itself) for visual vertices with far_n < 0.02 and
  far_c ≥ 0.10. A grid enters only if every subject has ≥ 30 such vertices in both directions; otherwise "too few vertices".

**B. Robustness of the HCP result (15 test subjects, settings fixed)**
- B1, B2. 1 mm rigid misregistration of the coarse data, applied to every method; 2.4 / 3.2 mm.
  Source `results/week4/hcp_eval/week4_hcp_eval.json`: model_shift1 − fmin(smoothing_shift1, upsample_smoothing_shift1).
- B3–B6. Model PSF assumed 15% narrower / wider than the operator's; 2.4 / 3.2 mm.
  Source `results/week3/week3_test_scores.json`: model_psf0.85 / model_psf1.15 − fmin(smoothing, upsample_smoothing).

**C. Second acquisitions; settings chosen on the other subjects of the pair type**
- C1. StudyForrest 3T → 7T, voxel ratio ≥ 2.5, 5 subjects (the pre-declared primary of the first analysis).
- C2. StudyForrest 7T → 7T, all pairs, 7 subjects.
  Source `results/week8/sf_cross/sf_cross_loso.json`, per-subject values of those tests.
- C3. ds002702 3T 2.4 mm → 7T GE 0.8 mm, field-map corrected, 2 subjects.
- C4. ds002702 7T SE 1.05 mm → 7T GE 0.8 mm, field-map corrected, 2 subjects.
  Source `results/week7/ds002702/summary/kay_cross_cv_results.json`: model − competitor (visual); high mixing from the
  record's "10-101%" bin.

## Checks before the table is used

- A1, A2 (all) reproduce the held-out results: 2.4 mm 13/15 and 3.2 mm 14/15 subjects better.
- A4–A6 (all) equal the week-7 table (+0.033, +0.029, +0.003); the visual set recomputed from the npz files equals
  `kay_results.json`, which validates the high-mixing computation.
- C1 equals the pre-declared result: +0.0141, 2/5, p = 0.844.

## Output

`results/week11/robustness_table.md` and `.json`; the table enters the manuscript (Section 4.8) and is copied there by script.
