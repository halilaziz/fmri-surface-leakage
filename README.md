# Where coarse fMRI mixes the cortex

Code, analysis plans and group leakage maps for the paper *Where coarse fMRI mixes the cortex: operator-predicted leakage of
surface mapping, and when correcting it helps* (Halil Aziz Velioğlu; preprint on bioRxiv, DOI to be added).

For each subject and acquisition, a forward operator built from the echo-planar point-spread function and the cortical surfaces
predicts how much of every vertex's standard (ribbon-constrained) surface value comes from cortex far away along the surface.
Inverting the operator under a Matérn surface prior corrects part of that mixing. The code reproduces the analyses of the paper
on four public datasets: HCP 7T retinotopy (with emulated coarser acquisitions), StudyForrest multi-resolution data, OpenNeuro
ds002702 and the NYU Retinotopy Dataset (OpenNeuro ds003787, 3T).

## Contents

| path | what |
|---|---|
| `src/` | the analysis code, one folder per analysis stage (below) |
| `theory_sims/` | sheet-sampling simulations behind the resolvability lookups |
| `run/` | `run_hcp.sh`, `run_studyforrest.sh`, `run_ds002702.sh`, `run_nyu.sh`: every step of each dataset's pipeline, in order, with its command; `run_statistics.sh`: the pre-specified tests, summary tables and result figures from the result files |
| `leakage_maps/` | group leakage maps (GIFTI): HCP 7T geometry on fs_LR 32k, and the NYU 3T acquisition on fsaverage (`3T_nyu/`); see its README |
| `plans/` | the analysis plans with the SHA-1 sums recorded before each analysis |
| `subjects/` | the subjects used (HCP: study codes only), subject-set hashes, the ds003787 file list |
| `ENVIRONMENT.md` | containers, versions, configuration variables, compute |

## Analysis stages

The plans and the result folders use the names the stages had during the study (stage N's code was in `src/weekN` and writes to
`results/weekN`); the folders here have descriptive names.

| folder | what it does | results | plan |
|---|---|---|---|
| `src/01_resolvability` | resolvable-wavelength lookups and per-subject resolvability maps (with `theory_sims/`) | `results/week1` |  |
| `src/02_forward_operator` | forward operator (effective PSF, column integration), Workbench ribbon weights, k-space emulation, leakage of standard mapping, operator checks | `results/week2` |  |
| `src/03_hcp_emulation` | the estimator (Matérn–SPDE posterior), competitors, HCP emulation fits and scoring, averaging kernels | `results/week3` |  |
| `src/04_misregistration_studyforrest` | HCP misregistration, depth and uncertainty tests; StudyForrest preprocessing and cross-session predictions | `results/week4` |  |
| `src/05_studyforrest_within_session` | StudyForrest per-run responses, within-session emulation, self-tuning | `results/week5` |  |
| `src/06_noise_dose_response` | noise spectra, correlated-noise likelihood, gain by far leak (dose-response) | `results/week6` |  |
| `src/07_ds002702_gate_group_maps` | ds002702 download, preprocessing, checks, emulation and cross-session tests; leak-gated estimator; HCP group leakage maps | `results/week7` |  |
| `src/08_contamination` | contamination-weighted leak (CWL) and the first pre-specified contamination analysis | `results/week8` | `STATS_PLAN.md` |
| `src/09_contamination_denoised` | the second pre-specified contamination analysis; concept figure | `results/week9` | `STATS_PLAN_w9.md` |
| `src/10_figure_data` | per-subject data for the figure panels | `results/week10` |  |
| `src/11_summary_table` | the pre-specified cross-dataset summary table | `results/week11` | `TABLE_SPEC.md` |
| `src/13_hcp_cohort` | the confirmatory HCP cohort: ring maps and fits, scoring, tests, figure | `results/week13` | `COHORT_PLAN.md` + amendment |
| `src/14_robustness_checks` | noise ceiling, polar-angle effect size, bin-convention sensitivity, subject-set checks (`verify_splits.py`) | `results/week14` |  |
| `src/15_far_part_tests` | far-part ablation, realistic native-acquisition kernels, the far part on real coarser StudyForrest sessions | `results/week15` | `ABLATION_PLAN.md`, `SF_OPERATOR_PLAN.md` |
| `src/16_sensitivity_analyses` | kernel-width scan, re-tuned half-run settings, correlated noise, far threshold and true geodesics | `results/week16` | `PLAN_BEFG.md` + addendum |
| `src/17_nyu_3t_replication` | NYU Retinotopy Dataset: preprocessing, emulation, the pre-specified replication, the coarse-data rule and registration analyses, 3T group maps | `results/week17` | `PLAN_NYU.md`, `PLAN_NYU_GATE.md` |
| `src/18_atlas_contamination` | contamination from an anatomical retinotopy atlas | `results/week18` | `PLAN_ATLAS.md` |

## Data (not included)

- **HCP 7T retinotopy** (Benson et al., 2018): ConnectomeDB / the `hcp-openaccess` S3 bucket, under the WU-Minn HCP Open Access
  Data Use Terms. Per subject the code needs the FIX-cleaned concatenated 7T retinotopy series (MNINonLinear) and its SBRef, the
  native-mesh surfaces and shape maps (MNINonLinear/Native, including the MSMAll sphere), and the per-run goodvoxels masks and
  motion regressors.
- **StudyForrest multi-resolution** (Sengupta et al., 2017, 2018): studyforrest.org (DataLad; ODC PDDL); 7T sessions at 0.8–3.0 mm,
  3T sessions at 1.4–3.0 mm, FreeSurfer reconstructions.
- **ds002702** (Kay et al., 2019, 2020): OpenNeuro, doi:10.18112/openneuro.ds002702.v1.0.1 (CC0), and the dataset authors'
  FreeSurfer reconstructions at osf.io/j2wsc; `src/07_ds002702_gate_group_maps/s3_openneuro.py` downloads from the public bucket.
- **NYU Retinotopy Dataset** (Himmelberg et al., 2021): OpenNeuro ds003787 (CC0), with its FreeSurfer reconstructions and stimulus
  apertures; `subjects/ds003787_subset_keys.json` lists the files used and `run/run_nyu.sh` downloads them.

**HCP restricted data and subject IDs.** HCP restricted information (family structure) was used only to assign the 20 development
subjects to 5 tuning and 15 test subjects, to exclude relatives of the tuning subjects from the confirmatory cohort, and to pick
one subject per family for the cohort's primary tests; no restricted data are in this repository. Because the analyses use
restricted data, HCP subject IDs are not published (HCP Restricted Data Use Terms): subjects are identified by study codes
(T01–T05 tuning, D01–D15 test; `subjects/hcp_mve_subjects.tsv`), and the key linking codes to HCP IDs is deposited with the HCP,
where HCP restricted-data users can obtain it. Set `HCP_CODE_KEY` to that key to run `run/run_hcp.sh`.
`subjects/subject_set_hashes.md` gives SHA-256 sums of the larger subject sets, to check with
`src/14_robustness_checks/verify_splits.py` and the HCP restricted family table. The confirmatory cohort (`plans/COHORT_PLAN.md`) is
every other HCP 7T retinotopy subject with a concatenated retinotopy run, minus seven (five relatives of the tuning subjects, two
with incomplete release packages); its list is not included, because with the one-per-family subset it would encode family
structure. `src/13_hcp_cohort/cohort_stats.py --family-rule lowest` needs the family table; without it, use `--family-rule none`.

## Configuration

The code reads its data roots from environment variables.

| variable | meaning |
|---|---|
| `GEO_ROOT` | project root holding `data/` (HCP in `data/hcp_extra`, StudyForrest in `data/studyforrest`) and `results/` |
| `GEO_KAY_ROOT` | workspace for ds002702 (`data/ds002702`, `data/ds002702_freesurfer`, `results/kay/…`) and the NYU data (`data/ds003787`, `results/nyu/…`) |
| `GEO_PROJECT` | folder holding `results/week17` for the NYU summaries |
| `HCP_CODE_KEY` | the study-code key (code, HCP ID, role), from the HCP |
| `TEMPLATEFLOW_HOME` | TemplateFlow home with `tpl-fsLR` (inside the XCP-D container by default) |
| `GEO_WB_LOCK` | lock file serialising Workbench volume-to-surface mapping, or `none` on file systems without locks |
| `WB_THREADS` | threads for Workbench calls |

## Running

Each `run/run_*.sh` lists every step in order with its command and outputs; run the steps one by one or adapt them to your
scheduler (the original runs used SLURM job arrays on an HPC cluster and Apptainer on a lab server). `run/run_statistics.sh`
then computes the pre-specified tests (`results/week8/w8_stats.md`, `results/week9/w9_stats.md`, `results/week13/cohort_final.md`,
`results/week17/nyu_test.md`, `results/week18/atlas_test.md`), the summary tables and the result figures. See `ENVIRONMENT.md`
for containers, versions and compute.

## Leakage maps

`leakage_maps/group_leakage.fsLR32k.{L,R}.func.gii` hold, for 1.6, 2.4 and 3.2 mm voxels (HCP 7T PSF), the mean far-leak fraction
of standard mapping over 19 subjects, its subject s.d. and the fraction of subjects above 10%. `leakage_maps/3T_nyu/` holds the same
for a typical 3T acquisition (NYU, 2.0 mm, and its 3.0 / 4.0 mm emulations) on fsaverage (164k): the mean far-leak fraction and
the share of subjects above 5%.

## Citation and licence

Please cite the paper (bioRxiv preprint; DOI to be added) and the datasets above. Licence: **[to be chosen]**.
