# Week 15, analysis D — the operator's far part tested on real coarser acquisitions (StudyForrest; written before running)

Section 4.1 validates the geometry of the operator's far part against *emulated* coarsening of the same HCP data. StudyForrest
has real sessions of the same subjects at 7T 0.8, 1.4, 2.0 and 3.0 mm (provider-distortion-corrected; registered to the
subject's FreeSurfer anatomy in week 4). Here the week-2 test is repeated across real sessions.

## Fixed before running

- **Subjects and sessions.** The 7 StudyForrest subjects, 7T sessions only; every ordered pair fine → coarse (0.8 → 1.4, 2.0,
  3.0; 1.4 → 2.0, 3.0; 2.0 → 3.0). Operators, ribbon weights, registrations and session grids as in week 4 (`sf_model.py`): the
  native acquisition kernel of each session, surfaces moved into each session's EPI space.
- **Field estimate.** As in week 2: the fine session's standard map divided by its partial-volume dilution (floor 0.3), on the
  unknowns covered by the fine session; no model fit is involved.
- **Predictors of the coarse session's standard map** (A_c = W_c K_c, split per vertex into near and far columns with the
  leakage-map definition: other hemisphere or > 15 mm on the area-scaled FreeSurfer sphere): *naive* (the fine map itself),
  *near only* (A_near f̂), *near + far* (A f̂), *near renormalised* (A_near f̂ scaled to the full row mass), *near + shuffled far*
  (the far contributions permuted across the vertices of the stratum, as in week 2).
- **Score and strata.** Residual energy fraction after the best real scaling per hemifield map (`resid_real`), on vertices
  covered by both sessions: all covered; visual cortex; visual cortex by the far leak of standard mapping at the coarse session
  (< 2%, > 5%, > 10%, > 20%); pericalcarine. A stratum counts with ≥ 50 vertices.
- **Predictions** (per pair type over subjects, and per subject pooled over its pairs; one-sided Wilcoxon where a direction is
  stated, n ≤ 7; the shuffled control calibrates the comparison):
  - D1: near + far beats near only where the predicted far leak is high (visual, leak > 10%): resid(full) < resid(near).
  - D2: the true far part beats a shuffled far part: resid(shuffled) > resid(full) in the same stratum.
  - D3: geometry, not mass: resid(full) < resid(renormalised near) in the same stratum.
  - Descriptive: the same over all visual vertices and where leak < 2% (expected: no difference).
- **Interpretation, stated in advance.** D1–D3 holding is the first validation of the far part on real coarser acquisitions
  (with the usual cross-session confounds raising every residual). D2 failing means the far part's geometry is not supported by
  real coarse data at these resolutions. Between-session differences in the pairs that involve 0.8 mm (slab coverage) may make
  those pairs uninformative; the 1.4 → 3.0 mm pair is the primary one.
- **Status.** Exploratory, reviewer-requested; hashed in `SF_OPERATOR_PLAN.sha1` before running.
