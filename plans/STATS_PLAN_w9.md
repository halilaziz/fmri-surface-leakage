# Week 9 — pre-declared plan: contrast-weighted leak with a denoised response estimate

Written 2026-10-01, before any week-9 quantity was computed. The week-8 results it refines are known: CWL-M was negative in HCP,
in the predicted direction but not significant in ds002702, and null in StudyForrest. The SHA-1 of this file is written into
every week-9 output.

## Question

Week 8's CWL used the source's own native-resolution map as the response estimate. In StudyForrest it failed, and the suspected
reason is noise in that estimate: response differences between far vertices are then partly noise. Does a denoised estimate
recover the dose-response?

## Definitions (as `results/week8/STATS_PLAN.md`, part A, except the response estimate)

- **S4 (primary):** the source's native-resolution standard map, smoothed on the surface with a 4 mm FWHM. Heat-kernel
  smoothing on each hemisphere's submesh of the unknowns is the same operator as the competitors' smoothing. Hemispheres are
  never mixed.
- **S2, S8 (secondary):** the same with 2 mm and 8 mm FWHM.
- **Q (secondary, practical):** the model's estimate from the coarse data, rendered at the native grid. It uses the setting of the
  week-8 analysis for that subject: ds002702 and StudyForrest leave-one-subject-out, HCP the week-3 choice. It needs no fine data.
- **Unchanged:** the far set, the readout rows, the normalisation by the typical response of the scored set, the bins (0–2,
  2–5, 5–10, 10–20, > 20%), the clean-at-native criterion, the ≥ 30 vertices rule, the gain Δ and the settings, all as in week 8.

## Tests

- **Primary:** D = Δ(highest bin with ≥ 30 vertices) − Δ(0–2% bin) per subject, using S4. Two datasets:
  - StudyForrest: 7 subjects; mean of the four session × factor cells per subject.
  - ds002702: 5 subjects; the ×3 (2.4 mm) cells.
  - H1: D < 0 in each. One-sided Wilcoxon signed-rank and a bootstrap 95% interval of the mean (10,000 resamples). No
    correction across the two datasets (independent), as in week 8.
- **Secondary (Holm over all):**
  - S2, S8 and Q in StudyForrest and ds002702 ×3;
  - S4 and Q in HCP 2.4 / 3.2 mm (consistency check: week-8 CWL-M was already negative there);
  - ds002702 ×2 and ×1.3125 with S4: descriptive only.
- **Paired comparison with week 8 (descriptive):** D(S4) − D(CWL-M) per subject.

A null primary result is reported as such. The exploratory other-hemisphere result of week 8 is not part of this plan.
