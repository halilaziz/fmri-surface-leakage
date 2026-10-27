# Week 8 — pre-declared analysis and statistics plan

Written 2026-09-30, before any week-8 quantity was computed: no contrast-weighted leak had been computed on any dataset, and
the StudyForrest cross-session re-analysis below had not been run. The weekly reports have already shown the far-leak
dose-response of HCP and StudyForrest (weeks 6–7) and the ds002702 results (week 7) descriptively. Tests on those earlier
numbers are therefore confirmatory in form only, and are marked as such. The SHA-1 of this file is written into every week-8
output.

## A. Contrast-weighted leak (CWL): does mixing of *different* responses predict the model's gain?

**Definitions.**
- Far set F(v): unknowns u in the other hemisphere, or more than 15 mm away on the registration sphere scaled to the cortical
  area (week 2).
- Readout row a_v = (W_c K_c)_v: standard ribbon mapping of the coarse-grid operator. tot_v = Σ_u a_vu.
- Response estimate x_u,j per vertex and map j (real maps; complex maps split into real and imaginary parts):
  - **M (mechanism):** the source's own native-resolution standard map. This is the source half (StudyForrest, ds002702) or the
    source run (HCP). It never uses target data.
  - **P (practical):** the source's coarse data, upsampled in k-space and mapped at the native grid. This needs no
    fine-resolution data.
- Contamination c_v,j = Σ_{u∈F(v)} a_vu (x_u,j − x_v,j) / tot_v: the change of the standard map at v caused by far cortex
  responding differently from v.
- **CWL_v = sqrt(Σ_j c_v,j²) / sqrt(mean over the scored set of Σ_j x_v,j²)**, i.e. contamination as a fraction of the typical
  response. The scored set is the same vertex set as the dose-response: visual (StudyForrest, ds002702) or coherent (HCP).

**Bins** (as the far-leak bins): 0–2%, 2–5%, 5–10%, 10–20%, > 20% of the typical response. Vertices must be clean at the
native grid (far leak < 2%). A bin is used for a subject if it has ≥ 30 vertices in each direction.

**Gain** Δ = residual(model) − residual(best competitor) in a bin (negative = model better). Settings are those of the
original analysis:
- HCP: the week-3 fixed choices (model 12 mm / 0.125 at 2.4 mm, 24 mm / 0.25 at 3.2 mm; competitor upsampling + 8 mm).
- StudyForrest: week-7 gate runs, leave-one-subject-out.
- ds002702: week-7 emulation, leave-one-subject-out.

**Primary contrast per dataset:** D = Δ(highest CWL-M bin available for the subject) − Δ(0–2% bin), one value per subject.
StudyForrest averages its four session × factor cells per subject; ds002702 uses the ×3 (2.4 mm) cells. HCP has two grids.
- H1 (predicted): D < 0 in each dataset.
- Test: one-sided Wilcoxon signed-rank over subjects (HCP n = 15, StudyForrest n = 7, ds002702 n = 5; with n = 5 the smallest
  attainable p is 0.031), plus a bootstrap 95% interval of the mean of D (10,000 resamples of subjects, percentile).
- Correction: none across datasets (independent). Holm across HCP's two grids.
- **The key prediction is ds002702:** there the far-leak contrast was flat or positive (week 7). H1 predicts D < 0 with
  CWL-M there too.

**Secondary (Holm-corrected together):**
- (a) The same contrast with far-leak bins, for comparison: confirmatory in form only.
- (b) CWL-P instead of CWL-M.
- (c) A CWL-P-gated estimator: model where CWL-P > τ, competitor elsewhere, τ ∈ {0, 1, 2, 5, 10, 20%, never} chosen
  leave-one-subject-out, model scaled onto the competitor without target data. Test: gated < competitor, one-sided
  signed-rank.

## B. StudyForrest cross-session selection at scale (the ds002702 week-7 3c rule)

Week-4 StudyForrest cross-session predictions, all settings as logged: competitor = source map ± smoothing (2–16 mm) or resampled
to the target grid + smoothing (2–16 mm); model = the week-4 prior grid (3–24 mm × 0.125–2). No field maps exist for the 3T
sessions, so there is no distortion correction (the 7T data are corrected by the providers).
- **Primary:** 3T → 7T pairs with a voxel ratio ≥ 2.5 (3T 2.0 mm → 7T 0.8 mm and 3T 3.0 mm → 7T 0.8 mm; 5 subjects). Each
  method's setting is chosen on the other subjects of the same pair type (leave-one-subject-out), visual vertices. Δ per subject =
  mean over the two pair types.
  - H2: Δ < 0 (the ds002702 3T → 7T gain generalises). One-sided signed-rank (n = 5) and bootstrap interval.
- **Secondary:** all 3T → 7T pairs; all 7T → 7T pairs; covered vertices.

## C. Reporting

Every test is reported with n, the mean and median of the per-subject values, the bootstrap interval and the p-value. A
non-significant primary result is reported as such, with no post-hoc subgroup rescue. Analyses added after this plan are marked
"exploratory".
