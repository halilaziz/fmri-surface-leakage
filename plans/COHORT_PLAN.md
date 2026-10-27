*Release copy: HCP subject IDs are replaced by study codes or withheld, as the HCP Restricted Data Use Terms require; the recorded SHA-1 is of the original file.*

# Week 13: full HCP 7T retinotopy cohort — analysis plan (hashed before any fit of a new subject)

## What is fixed before the data are seen

- **Subjects.** All HCP 7T retinotopy subjects not in the pilot (164), minus 5 relatives of the 5 tuning subjects and 2 whose
  S3 packages are incomplete ([HCP ID withheld]: no concatenated retinotopy run; [HCP ID withheld]: no MSMAll spheres): **158 subjects in 86
  families**. The 15 pilot test subjects were used to develop the two new analyses below and are reported separately.
- **Settings.** Everything is frozen at the week-3 choices: model prior 2.4 mm (12 mm, 0.125), 3.2 mm (24 mm, 0.25);
  competitors as chosen on the 5 tuning subjects (evaluate.py re-derives them from the stored tuning estimates); the
  contamination measures as in the first and second pre-declared analyses. No setting is tuned on any new subject.
- **Pipeline per subject.** Week-1 maps, week-2 operator / leakage / wedge maps, ring maps (week 13), week-3 fits (wedge; PSF
  stretched 0.85 / 1.15; wedge from the first and from the second half of each source run; ring), contamination maps
  (CWL-M/P, S2/S4/S8/Q for the wedges; CWL-M/P for the rings).
- **Unit of analysis and family structure.** The subject. Twins and siblings are not independent, so the **primary tests use
  one subject per family: the lowest subject ID of each family** (86 subjects). Secondary: all 158 subjects, with the same
  tests and, in addition, a family-level bootstrap (resample families) for the interval of the mean.
- **Inclusion per test.** A subject enters a test when every stratum the test needs has ≥ 30 vertices (as in weeks 3–9). A
  subject whose pipeline fails a check that the pilot pipeline also applied (operator column mass, coherent-vertex count > 0)
  is excluded from all tests and listed.
- **Statistics.** As in the pre-declared analyses: one-sided Wilcoxon signed-rank across subjects (direction stated per
  hypothesis), bootstrap 95% interval of the mean (10,000 resamples, seed 0), subjects in the predicted direction. Secondary
  tests Holm-corrected within their family of tests. The pilot results are not combined with the new cohort.

## Hypotheses

Notation: Δ = residual(model) − residual(competitor) (negative: model better). Unless stated, the competitor is the one of the
pre-declared analyses, upsampling + 1.6 mm mapping + 8 mm smoothing. Strata by the dose variable at the coarse grid among
vertices with < 2% far leak at 1.6 mm; bins 0–2, 2–5, 5–10, 10–20, > 20%; D_x = Δ(highest bin with ≥ 30 vertices) − Δ(0–2% bin)
for the dose variable x (far leak, CWL-M, CWL-P, CWL-S4, CWL-Q), computed by `summarize_cwl.py` / `summarize_cwl2.py` exactly
as in `STATS_PLAN.md`. "All coherent" gains use the per-subject better of the two smoothing competitors, as in Section 4.8.
Sources of the numbers per batch: `eval/hcp_cwl_summary.json`, `eval/hcp_cwl2_summary.json`, `eval/ring_cwl_summary.json`
(ring), `eval/dose_*/hcp_dose_response.json` (far-leak bins, competitor upsampling + 8 mm smoothing) and
`eval/*/week3_test_scores.json` (all-coherent and PSF rows).

**H1 — confirmatory replication (wedges).** At 2.4 and 3.2 mm, D_CWL-M < 0 (primary, as the first analysis' primary) and
D_CWL-P < 0 (the coarse-data-only measure). Four tests; the primary family is {2.4 mm CWL-M, 3.2 mm CWL-M}.
Secondary: D_far < 0; the overall gain Δ(all coherent) < 0; the subject counts and sizes of Section 4.3 restated.

**H2 — rings (same subjects, same acquisition).** The expectation before the pilot was that rings, which drive both banks of
the calcarine alike, would show no far-leak dose-response. The exploratory look at the 15 pilot subjects (recorded below) did not
bear this out: the ring runs have a far-leak dose-response of their own at 2.4 mm, as strong as the wedges', and a weaker one at
3.2 mm, and contamination predicts the ring gain at both grids. Eccentricity differs across much of the far cortex (foveal and
peripheral representations are far apart on the sheet), so the far part of the kernel does carry different responses for rings
too. The confirmatory hypotheses are therefore:
- H2a: at 3.2 mm, D_far(wedge) − D_far(ring) < 0 (the plain-leak dose-response is steeper for the stimulus with contrast across
  sulcal banks; pilot −0.016, 15/15). At 2.4 mm the same difference is reported with its interval, without a directional
  prediction (pilot −0.001, 8/15).
- H2b: D_CWL-M(ring) < 0 at 2.4 and 3.2 mm (contamination predicts the gain for rings as for wedges; pilot −0.028 and −0.027,
  13/15 each).
- H2c (secondary): at 3.2 mm, D_CWL-M(ring) − D_far(ring) < 0 (for rings, contamination separates the vertices where the model
  helps more sharply than the plain leak; pilot −0.027 against −0.007).
- Descriptive: D_far(ring) with its interval at both grids; the model's overall gain for rings (pilot −0.005 and −0.012).

**H3 — source noise inside one dataset.** Fits from half a source run (4 cycles) against the same full-run reference. The
expectation before the pilot was that, if part of the model's gain were denoising, the gain where no unmixing is needed would
grow with source noise. The exploratory look showed the opposite direction: the gain in the 0–2% far-leak bin shrank with half
runs (2.4 mm −0.0086 → −0.0045, 14 of 15 subjects shrank; 3.2 mm −0.0165 → −0.0140, 13 of 15), while the leak-dependent part of
the gain did not change. The confirmatory hypotheses are therefore:
- H3a: Δ_clean(half) − Δ_clean(full) > 0 at 2.4 and 3.2 mm (the model's gain where no unmixing is needed decreases with source
  noise: it is not a denoising gain), where Δ_clean = Δ in the 0–2% far-leak bin of the far-leak dose-response (competitor
  upsampling + 8 mm smoothing) and "half" is the mean of the first- and second-half fits (one-sided).
- H3b: D_far(half) − D_far(full) (same source): the 95% interval includes 0 (the leak-dependent gain is unchanged). Reported as an
  interval, not a significance test.
- H3c (secondary): Δ_clean(half) < 0 at both grids (the model still beats the competitor with half the data; pilot 10/15 at
  2.4 mm, 15/15 at 3.2 mm).
- Descriptive: the noise ceiling (split-half agreement) of half- and full-run sources.

**H4 — PSF sensitivity (from Section 4.8, now with n ≈ 86).** With the PSF assumed 15% too wide, Δ(all coherent) < 0 at
2.4 mm (the pilot's p = 0.23 case); with 15% too narrow, Δ(all coherent) < 0 at both grids (secondary). Competitor: the
per-subject better of the two smoothing competitors, as in the rows B3–B6 of Section 4.8.

## What the plan does not cover

- The leak-gated estimator, the averaging-kernel and known-truth analyses of week 3 (not recomputed for the cohort).
- Group maps of the gain on fs_LR (descriptive; produced from the same fits, no test).
- Bars (RETBAR1/2): not periodic, would need pRF fitting; not analysed.

## Changes after hashing

None are allowed; a change found necessary is recorded as a deviation with its reason, as in week 11.

## Record of the exploratory look (15 pilot test subjects, before hashing)

`results/week13/pilot_explore.md` / `.json`, from `results/week13/cohort/pilot/eval/`. The pilot pipeline reproduced the
published pilot numbers (D_CWL-M −0.023 / −0.037 at 2.4 / 3.2 mm, 11/15 and 15/15; overall gain −0.007 / −0.013). New:

| quantity (pilot, exploratory) | 2.4 mm | 3.2 mm |
|---|---|---|
| D_far(ring) | −0.024 (13/15) | −0.007 (10/15) |
| D_far(wedge) − D_far(ring) | −0.001 (8/15) | −0.016 (15/15) |
| D_CWL-M(ring) | −0.028 (13/15) | −0.027 (13/15) |
| ring: overall gain | −0.005 (12/15) | −0.012 (14/15) |
| Δ_clean(full) → Δ_clean(half) | −0.0086 → −0.0045 | −0.0165 → −0.0140 |
| Δ_clean(half) − Δ_clean(full) | +0.0041 (14/15 > 0) | +0.0024 (13/15 > 0) |
| D_far(half) − D_far(full) | +0.001 [−0.003, +0.004] | −0.000 [−0.003, +0.002] |

H2 and H3 above were rewritten after this look; H1 and H4 were not changed. The pilot subjects are not part of the
confirmatory cohort.

- Plan hash: `COHORT_PLAN.sha1` (SHA-1 and time), written before the first fit of any of the 158 subjects.
