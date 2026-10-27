#!/bin/bash
# HCP 7T retinotopy: every step of the paper's HCP analyses, in order. Run step by step (each loop can be parallelised per
# subject; the original runs used SLURM job arrays). Output folders under $GEO_ROOT/results are fixed by the scripts.
#   GEO_ROOT  project root with data/hcp_extra/<subject>/... (HCP files, see README) and results/
#   XCPD      path to xcp_d.sif (pennlinc/xcp_d:26.0.2)
#   HCP_CODE_KEY  the study-code key (code, hcp_id, role); HCP subject IDs are not published (HCP Restricted Data Use Terms),
#                 the key is deposited with the HCP and available to HCP restricted-data users
set -euo pipefail
: "${GEO_ROOT:?set GEO_ROOT}"; : "${XCPD:?set XCPD to xcp_d.sif}"
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/src; R=$GEO_ROOT/results
PY="apptainer exec --bind $GEO_ROOT --env GEO_ROOT=$GEO_ROOT $XCPD python3"
: "${HCP_CODE_KEY:?set HCP_CODE_KEY to the study-code key (from the HCP)}"
ALL=$(awk -F'\t' 'NR>1 {print $2}' "$HCP_CODE_KEY")
TUNE=$(awk -F'\t' 'NR>1 && $3=="tuning" {print $2}' "$HCP_CODE_KEY")
TEST=$(awk -F'\t' 'NR>1 && $3=="test" {print $2}' "$HCP_CODE_KEY")
step() { echo; echo "=== $*"; }

step "week 1: resolvability (theory simulations, lookup tables, per-subject maps, summary)"
(cd $HERE/theory_sims && for t in run_experiments run_banks_orient run_banks_realistic make_figures; do $PY $t.py; done)   # -> theory_sims/results
$PY $C/01_resolvability/lookup_sims.py $R/week1/lookup --workers 14
for S in $ALL; do $PY $C/01_resolvability/surface_maps.py $S $R/week1/lookup $R/week1/maps; done
$PY $C/01_resolvability/summarize.py $R/week1/maps $R/week1/lookup $HERE/theory_sims/results

step "week 2: forward operator, HCP ribbon weights, leakage maps, emulated 2.4 / 3.2 mm, cross-resolution prediction"
for S in $ALL; do $PY $C/02_forward_operator/run_subject.py $S $R/week2/subjects $R/week1/maps --threads 8; done
$PY $C/02_forward_operator/test_checks.py                                       # brute-force and emulation-consistency checks (2 subjects)
$PY $C/02_forward_operator/summarize.py $R/week2/subjects $R/week1 $R/week2/_summary
$PY $C/10_figure_data/w2_paper_extract.py $R/week2/subjects $R/week1 $R/week2/paper_extract   # small inputs of the paper's Fig. 1B / 2

step "week 3: model and competitors on the emulated grids; tuning on 5 subjects, then the 15 held-out subjects"
SET="3:1,3:0.5,3:0.25,3:0.125,6:1,6:0.5,6:0.25,6:0.125,12:1,12:0.5,12:0.25,12:0.125,24:1,24:0.5,24:0.25,24:0.125"
SET_EXT="6:0.0625,6:0.03125,12:0.0625,12:0.03125,24:0.0625,24:0.03125"                 # extension towards stronger priors
for S in $TUNE; do
  $PY $C/03_hcp_emulation/run_subject.py $S $R/week3/tune --settings $SET --grids 2.4,3.2 --threads 10
  $PY $C/03_hcp_emulation/run_subject.py $S $R/week3/tune_ext --settings $SET_EXT --grids 2.4,3.2 --threads 10
done
$PY $C/03_hcp_emulation/evaluate.py $R/week3/tune,$R/week3/tune_ext $R/week3/test_all $R/week3/_eval --tune-only   # -> chosen settings
W3SET='2.4=24:0.25,12:0.125;3.2=24:0.25'        # chosen on the tuning subjects (the last setting per grid gets kernels/simulation)
for S in $TEST; do
  $PY $C/03_hcp_emulation/run_subject.py $S $R/week3/test_all --settings "$W3SET" --grids 2.4,3.2 --akernels 24 --simulate --threads 10
  for a in 0.85 1.15; do $PY $C/03_hcp_emulation/run_subject.py $S $R/week3/test_psf$a --settings '2.4=12:0.125;3.2=24:0.25' --grids 2.4,3.2 --psf-stretch $a --threads 10; done
done
$PY $C/03_hcp_emulation/evaluate.py $R/week3/tune,$R/week3/tune_ext $R/week3/test_all,$R/week3/test_psf0.85,$R/week3/test_psf1.15 $R/week3/_eval
$PY $C/03_hcp_emulation/summarize.py $R/week3/_eval $R/week3/test_all $R/week3/_summary

step "week 4: misregistration (model and competitors), depth component, uncertainty"
for S in $TEST; do for M in shift shiftcomp depth uncertainty; do $PY $C/04_misregistration_studyforrest/run_hcp.py $S $R/week4/hcp $M --grids 2.4,3.2 --threads 8; done; done
$PY $C/04_misregistration_studyforrest/evaluate_hcp.py $R/week4/hcp/_eval
$PY $C/04_misregistration_studyforrest/summarize_hcp.py $R/week4/hcp/_eval/week4_hcp_eval.json $R/week4/hcp/_summary

step "week 6: dose-response of the gain on far leak"
$PY $C/06_noise_dose_response/hcp_dose_response.py $R/week6/hcp

step "week 7: leak-gated estimator, group leakage maps"
$PY $C/07_ds002702_gate_group_maps/hcp_gate.py $R/week7/hcp
$PY $C/07_ds002702_gate_group_maps/leak_maps.py $R/week7/leakage_maps

step "week 8: contamination-weighted leak (pre-declared, plans/STATS_PLAN.md)"
SHA8=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN.sha1" | head -1)
for S in $TEST; do $PY $C/08_contamination/cwl_hcp.py $S --plan-sha1 $SHA8; done
$PY $C/08_contamination/summarize_cwl.py hcp $R/week8/cwl_hcp $R/week8/summary/hcp_cwl_summary.json

step "week 9: contamination with denoised response estimates (pre-declared, plans/STATS_PLAN_w9.md)"
SHA9=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN_w9.sha1" | head -1)
for S in $TEST; do $PY $C/09_contamination_denoised/cwl2_hcp.py $S --plan-sha1 $SHA9; done
$PY $C/09_contamination_denoised/summarize_cwl2.py hcp $R/week9/cwl2_hcp $R/week9/summary/hcp_cwl_summary.json
echo "HCP done; the tests and figures are built by run_paper.sh"
# The confirmatory cohort (manuscript Sections 3.9 / 4.9) runs this same chain per subject, plus the ring maps/fits
# (src/13_hcp_cohort/ring_subject.py; --stimulus ring), the half-run fits (--segment half1|half2) and the PSF ±15% fits, through the
# SLURM job arrays (stages 0, A, B per subject, then scoring per batch; fits were refused until plans/COHORT_PLAN.sha1 was
# present at results/week13/); run_statistics.sh runs the tests.
