# Week 17 — External replication of the HCP emulation result in the NYU Retinotopy Dataset (written before any fit)

## Question

Do the model's advantage over the best competitor and its growth with far leak and with contamination, found in HCP 7T
emulation (15 development subjects; 158-subject confirmatory cohort), replicate in an independent dataset that differs in
site, scanner (Siemens Prisma), field strength (3T), voxel size (2.0 mm), stimulus (sweeping bars instead of rotating wedges)
and preprocessing (ours, from the raw data)?

## Data

OpenNeuro ds003787 v1.0.0 (NYU Retinotopy Dataset; Himmelberg, Kurzawski, Benson, Pelli, Carrasco, Winawer; CC0): 44 subjects,
4–12 bar-mapping runs each (192 volumes, TR 1 s, 2.0 mm isotropic, multiband 6, phase encoding P→A, effective echo spacing
0.58 ms, 104 phase-encoding lines, no partial Fourier), one spin-echo AP/PA field-map pair per session, FreeSurfer 6
reconstructions (provided), and one aperture sequence (100 × 100 × 192, binary; identical in every run). Downloaded to hebb
(`GEO_CONST/data/ds003787`, 97.3 GB subset: raw data, FreeSurfer, apertures, ROIs, the authors' pRF maps); nothing is written
outside `/analysis/halilaziz/halil/GEO_CONST`.

**Split, fixed now:** `numpy.random.default_rng(17).permutation` over the 44 sorted subject IDs; the first 8 are the tuning
subjects (sub-wlsubj057, 062, 084, 087, 094, 118, 121, 126), the other 36 are the test subjects (`nyu_split.json`).

**Exclusions (decided before any result):** a run is excluded if its maximum absolute translation relative to the reference
exceeds 4 mm (two voxels); a subject is excluded if fewer than 4 runs remain, if no field-map pair is usable, or if the
boundary-based registration fails in every start (cost > 0.8). Excluded runs and subjects are listed before any test-subject
result is summarised.

## Preprocessing (per subject; FSL tools in the fMRIPrep container on hebb)

1. topup on the AP/PA spin-echo pair (acquisition parameters from the sidecars) → field;
2. motion correction of every run to the run-1 single-band reference, itself registered rigidly to the P→A spin-echo image, in
   one spline resampling into the field-map grid (same 2.0 mm geometry);
3. distortion correction of every run with applytopup (P→A line, Jacobian modulation);
4. registration of the corrected session mean to the FreeSurfer anatomy: FLIRT from three starts, then boundary-based
   registration with the FreeSurfer white matter, lowest cost kept (as for ds002702);
5. brain mask as for ds002702.

## Polar-angle maps (the analogue of the HCP wedge Fourier coefficient)

Per run and grid, a GLM with three stimulus regressors — the aperture's coverage of the stimulated field weighted by 1,
cos θ and sin θ (θ = polar angle of each aperture pixel about the centre) — each convolved with the SPM double-gamma HRF,
plus the temporal derivative of the first, Legendre drifts of order 0–3 and the six motion parameters. In the aperture
sequence the three stimulus regressors are uncorrelated (|r| < 0.01). The complex map z = β_cos + i β_sin (percent signal
change) is the first angular harmonic of each voxel's response: its phase is a linear estimate of the polar angle, as the
phase of the wedge Fourier coefficient is in HCP. Noise variance of z per voxel: σ²_res · [(XᵀX)⁻¹_cos,cos + (XᵀX)⁻¹_sin,sin],
with σ²_res the GLM residual variance in percent signal change. Halves: odd and even runs (A, B); z averaged within a half,
variance divided by the number of runs.

## Emulation, operators, methods

- **Emulation:** every corrected time series is coarsened frame by frame to 3.0 mm (×1.5) and 4.0 mm (×2.0) with the k-space
  crop and Tukey window (α = 0.5) used for HCP, and the GLM is rerun on the coarse series.
- **Operators:** the FreeSurfer-anatomy operator of the ds002702 analysis (`kay_emul.py`: unknowns on the white–pial columns,
  Workbench ribbon weights per grid), with the acquisition PSF from the run sidecars (effective echo spacing, phase-encoding
  lines and axis; T2* 45 ms at 3T; taper 0.3, slice edge 0.15, interpolation blur 0.3 voxel) and the coarse kernels by the
  HCP emulation rule. Far leak and contamination exactly as in the paper (other hemisphere, or > 15 mm on the area-scaled
  registration sphere).
- **Prediction:** source half coarsened, target = the other half's native 2.0 mm standard map; both directions, averaged.
  Methods: standard mapping of the coarse data; + surface smoothing (FWHM 2, 4, 6, 8, 10, 12, 14, 16 mm); k-space upsampling
  to 2.0 mm; upsampling + smoothing (same FWHMs); the model (white-noise likelihood, Matérn prior; range r ∈ {6, 12, 24, 48} mm
  × amplitude s ∈ {0.0625, 0.125, 0.25, 0.5} × σ0, σ0 as for ds002702), rendered at 2.0 mm.
- **Score:** the HCP residual — energy of the target left after the best complex scaling of the prediction — per vertex set,
  averaged over the two directions.

## Vertex sets

- **responsive:** vertices whose run-averaged native 2.0 mm standard-mapped time series (all runs) is explained by the three
  stimulus regressors with R² ≥ 0.2 (after drifts and motion); the analogue of HCP's coherent vertices.
- **clean, leak > 10%:** responsive, far leak < 2% at 2.0 mm, far leak > 10% at the coarse grid.
- **pericalcarine:** responsive and in the FreeSurfer pericalcarine label.
- **Dose bins:** responsive vertices with < 2% far leak at 2.0 mm, binned by the dose variable at the coarse grid (0–2, 2–5,
  5–10, 10–20, > 20%); a subject enters a bin with ≥ 30 vertices; D_x = Δ(highest bin with ≥ 30 vertices) − Δ(0–2% bin).
  Dose variables: far leak; CWL-M (contamination with the source half's native 2.0 mm map as the response estimate,
  normalised as in the paper).

## Selection (tuning subjects only)

The week-3 rule: per grid, the model setting, the smoothing FWHM and the upsampling + smoothing FWHM with the lowest mean
residual over responsive vertices of the 8 tuning subjects. The **competitor** of the tests is the better of the two chosen
competitors on the tuning subjects. The choices are written to `SETTINGS_NYU.txt` before any test subject is fitted. The
tuning subjects' runs may be used to find and fix implementation errors; the test subjects are not looked at until then.

## Hypotheses (36 test subjects; Δ = residual(model) − residual(competitor), negative favours the model)

Primary, at both coarse grids (3.0 mm, ×1.5; 4.0 mm, ×2.0), one-sided Wilcoxon signed-rank, Holm across the six tests:
- **N1** — Δ over responsive vertices < 0 (the overall advantage replicates);
- **N2** — D_far < 0 (the gain grows with far leak);
- **N3** — D_CWL-M < 0 (the gain grows with contamination).

Secondary (Holm within the family): Δ(clean, leak > 10%) < 0; Δ(pericalcarine) < 0. Descriptive: the polar-angle error in
degrees (mean absolute angular difference between the phases of prediction and target over responsive vertices); the far-leak
figures of a 3T 2.0 mm acquisition and its 3.0 / 4.0 mm emulations (mean far leak, share of cortex above 5% and 10%,
all included subjects). Statistics as in the confirmatory cohort: subjects in the predicted direction, mean, bootstrap 95%
interval (10,000 resamples, seed 0), one-sided p.

## Decision rules

- N1 holds at both grids (Holm): the paper states that the HCP emulation result replicates at another site, scanner, field
  strength, voxel size and stimulus. N1 fails at a grid: the paper states that it did not replicate there, with the numbers,
  and the abstract and Discussion are revised accordingly.
- N2 / N3: replicated or not, stated per grid as above.
- Whatever the outcome, the analysis is reported in the manuscript (a new results subsection and a supplement section) and
  the release.

## Deviations

Any change to this plan (a failed preprocessing step, a changed threshold, an implementation fix that alters a definition)
is written into `DEVIATIONS_NYU.md` with its reason and time before the test subjects' results are summarised.

Hash in `PLAN_NYU.sha1`, recorded before the first fit.
