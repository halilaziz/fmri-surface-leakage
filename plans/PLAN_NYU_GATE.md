# Week 17b — three further analyses of the NYU Retinotopy data (written before any of them is computed)

User's decision (2026-10-07): "do items 2 and 3" — trim the manuscript and run three analyses on the existing NYU data. The
NYU results of PLAN_NYU.md are known (the overall gain replicated at 4.0 mm, not at 3.0 mm; contamination predicted the gain
at both grids, plain far leak did not); these analyses are added afterwards and are labelled as such. Settings, competitor,
subjects, split and scoring are those of PLAN_NYU.md (model 12 mm / 2 at 3.0 mm and 6 mm / 1 at 4.0 mm; competitor
upsampling + 2 mm smoothing; 8 tuning, 35 included test subjects).

## A. A practical rule: correct only where coarse-data contamination is high (pre-specified tests)

- **Measure.** CWL-P, the contamination-weighted leak of standard mapping with the response estimate taken from the coarse
  data alone (the source half's coarse map, k-space upsampled and mapped at 2.0 mm), exactly as in the paper's CWL-P.
- **Gated estimator.** Per vertex, the model's estimate where CWL-P > τ and the competitor's elsewhere, with τ ∈ {0, 0.02,
  0.05, 0.10, 0.20, ∞} (τ = 0: the model everywhere; τ = ∞: the competitor everywhere).
- **Choice.** Per grid, τ with the lowest mean residual over responsive vertices of the 8 tuning subjects.
- **Tests (35 test subjects, Δ over responsive vertices, one-sided Wilcoxon, Holm across the four):** G1: gated − competitor
  < 0 at 3.0 mm and at 4.0 mm; G2: gated − model < 0 at 3.0 mm and at 4.0 mm. If G1 and G2 hold at a grid, the rule beats both
  pure strategies there.
- **Descriptive:** D_CWL-P (the dose contrast with CWL-P bins, as D_CWL-M), the share of responsive vertices sent to the model,
  and the gated estimator's polar-angle error.

## B. Registration and the pericalcarine shortfall (descriptive)

Per test subject: the boundary-based registration cost, the 95th percentile of the absolute field (Hz) over the brain and
the median absolute field at the responsive pericalcarine vertices, against Δ over responsive vertices and Δ in the
pericalcarine (model − competitor). Spearman correlations (two-sided, n = 35) and a split at the median registration cost.
No test is confirmatory.

## C. Leakage maps of the 3T 2.0 mm acquisition as a resource (descriptive)

Per included subject, the far-leak maps of standard mapping at 2.0 mm and at the 3.0 / 4.0 mm emulations, carried from the
subject's FreeSurfer registration sphere to fsaverage (164k) by nearest-neighbour interpolation on the sphere (three nearest
vertices, inverse-distance weights); group mean and share of subjects above 5% per vertex; region table (FreeSurfer aparc,
area-weighted mean far leak and share above 10%, mean over subjects), as the HCP table of Section 4.2. Released as GIFTI.

## Compute

All on hebb inside `/analysis/halilaziz/halil/GEO_CONST`: `nyu_emul.py --gate` reruns every included subject at the chosen
settings (reproducing the published scores as a check) and adds CWL-P, the gated scores, the field statistics and the maps;
`nyu_groupmaps.py` builds C. Results copied to `results/week17/`.

Hash in `PLAN_NYU_GATE.sha1`, recorded before the first run.
