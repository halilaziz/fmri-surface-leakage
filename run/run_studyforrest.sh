#!/bin/bash
# StudyForrest multi-resolution: every step of the paper's StudyForrest analyses, in order.
#   GEO_ROOT  project root with data/studyforrest/{multires7t,multires3t,freesurfer} and results/
#   XCPD, FMRIPREP  paths to xcp_d.sif and fmriprep.sif; FS_LICENSE  FreeSurfer licence (bbregister)
set -euo pipefail
: "${GEO_ROOT:?}"; : "${XCPD:?}"; : "${FMRIPREP:?}"; : "${FS_LICENSE:?}"
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/src; R=$GEO_ROOT/results
PY="apptainer exec --bind $GEO_ROOT --env GEO_ROOT=$GEO_ROOT $XCPD python3"
PYFSL="apptainer exec --bind $GEO_ROOT --env GEO_ROOT=$GEO_ROOT,FS_LICENSE=$FS_LICENSE $FMRIPREP python"
SUBJ=$(awk 'NR>1 {print $1}' "$HERE/subjects/studyforrest_subjects.tsv")
step() { echo; echo "=== $*"; }

step "week 4: preprocessing, registration, cross-session predictions (all sessions, all source -> target pairs)"
for S in $SUBJ; do
  $PYFSL $C/04_misregistration_studyforrest/sf_prep.py $S
  $PYFSL $C/04_misregistration_studyforrest/sf_reregister.py $S
  $PY $C/04_misregistration_studyforrest/sf_fixmask.py $S
  $PY $C/04_misregistration_studyforrest/sf_model.py $S --threads 8 --prior-grid
done
$PY $C/04_misregistration_studyforrest/sf_summarize.py $R/week4/sf/model $R/week4/sf/_summary

step "week 5: per-run betas, within-session emulation (7T 1.4 / 2.0 mm, x1.5 / x2), source-only self-tuning"
for S in $SUBJ; do
  $PYFSL $C/05_studyforrest_within_session/sf_runbetas.py $S --sessions 7T-r14,7T-r20
  for ses in 7T-r14 7T-r20; do $PY $C/05_studyforrest_within_session/sf_emul.py $S $ses --threads 8; done
  $PY $C/05_studyforrest_within_session/sf_selftune.py $S
done
$PY $C/05_studyforrest_within_session/summarize_w5.py $R/week5/sf_emul $R/week5/sf_selftune $R/week4/sf/model $R/week5/_summary

step "week 6: noise spectra and the correlated-noise likelihood"
for S in $SUBJ; do for ses in 7T-r14 7T-r20; do
  $PY $C/06_noise_dose_response/noise_spectrum.py $S $ses $R/week6/noise
  for f in 1.5 2; do $PY $C/06_noise_dose_response/sf_emul_corr.py $S $ses --factor $f; done
done; done
$PY $C/06_noise_dose_response/summarize_w6.py $R/week6/sf_emul_corr $R/week6/_summary

step "week 7: per-vertex predictions for the leak-gated estimator; week-7 summary (with HCP and the leakage maps)"
for S in $SUBJ; do for ses in 7T-r14 7T-r20; do for f in 1.5 2; do $PY $C/07_ds002702_gate_group_maps/sf_gate.py $S $ses --factor $f; done; done; done
$PY $C/07_ds002702_gate_group_maps/summarize_w7.py $R/week7/hcp/hcp_gate.json $R/week7/sf_gate $R/week7/leakage_maps/group_leakage_summary.json $R/week7/_summary

step "week 8: contamination-weighted leak; cross-session selection on the other subjects (both pre-declared)"
SHA8=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN.sha1" | head -1)
for S in $SUBJ; do for ses in 7T-r14 7T-r20; do for f in 1.5 2; do $PY $C/08_contamination/cwl_sf.py $S $ses --factor $f --plan-sha1 $SHA8; done; done; done
$PY $C/08_contamination/summarize_cwl.py sf $R/week7/sf_gate $R/week8/cwl_sf $R/week8/summary/sf_cwl_summary.json
$PY $C/08_contamination/sf_cross_loso.py $R/week4/sf/model "$HERE/plans/STATS_PLAN.sha1" $R/week8/sf_cross     # reads the week-4 pair logs

step "week 9: contamination with denoised response estimates (model setting per cell: the week-8 leave-one-subject-out choice)"
SHA9=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN_w9.sha1" | head -1)
python3 - "$R/week8/summary/sf_cwl_summary.json" > /tmp/sf_cwl2_jobs.txt <<'EOF'
import json, sys
for cell, c in json.load(open(sys.argv[1]))['cells'].items():
    key, fac = cell.rsplit('_x', 1)
    for S, r in c.items():
        print(S, key, fac, r['model'])
EOF
while read S ses f m; do $PY $C/09_contamination_denoised/cwl2_sf.py $S $ses --factor $f --model $m --plan-sha1 $SHA9; done < /tmp/sf_cwl2_jobs.txt
$PY $C/09_contamination_denoised/summarize_cwl2.py sf $R/week7/sf_gate $R/week9/cwl2_sf $R/week9/summary/sf_cwl_summary.json
echo "StudyForrest done; the tests and figures are built by run_paper.sh"
