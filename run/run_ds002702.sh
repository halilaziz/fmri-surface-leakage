#!/bin/bash
# OpenNeuro ds002702 (Kay et al.): every step of the paper's ds002702 analyses, in order.
#   GEO_KAY_ROOT  workspace: data/ds002702 (BIDS), data/ds002702_freesurfer/subjNN (osf.io/j2wsc), results/kay/...
#   XCPD, FMRIPREP  paths to xcp_d.sif and fmriprep.sif
#   GEO_WB_LOCK=none on file systems without lock support (then run one Workbench mapping job per node)
set -euo pipefail
: "${GEO_KAY_ROOT:?}"; : "${XCPD:?}"; : "${FMRIPREP:?}"
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/src; G=$GEO_KAY_ROOT; K=$G/results/kay
ENV="GEO_KAY_ROOT=$G,GEO_WB_LOCK=${GEO_WB_LOCK:-none}"
PY="apptainer exec --bind $G --env $ENV $XCPD python3"
PYFSL="apptainer exec --bind $G --env $ENV $FMRIPREP python3"
step() { echo; echo "=== $*"; }
SES="sub-01:ring7T sub-02:ring7T sub-03:ring7T sub-04:ring7T sub-05:ring7T sub-01:ring3T sub-03:ring3T sub-01:ringSE sub-04:ringSE"

step "download (public S3 bucket; ~79 GB) and the authors' FreeSurfer reconstructions (osf.io/j2wsc, manual download)"
python3 $C/07_ds002702_gate_group_maps/s3_openneuro.py list ds002702 $G/data/ds002702_keys.json
python3 $C/07_ds002702_gate_group_maps/s3_openneuro.py get ds002702 $G/data/ds002702_keys.json $G/data/ds002702 --jobs 8

step "preprocessing per session (motion correction, FLIRT + BBR registration, per-run GLM), then QC"
for x in $SES; do $PYFSL $C/07_ds002702_gate_group_maps/kay_prep.py ${x%:*} ${x#*:}; done
$PY $C/07_ds002702_gate_group_maps/kay_betaqc.py $K/prep                                  # per-run betas: large-beta count, run-to-others correlation
for x in $SES; do $PY $C/07_ds002702_gate_group_maps/kay_regcheck.py ${x%:*} ${x#*:}; done > $K/regcheck_stage1.log

step "stage 2: field-map correction; per-session choice by the registration check"
for x in sub-01:ring7T sub-03:ring7T sub-04:ring7T sub-01:ring3T sub-03:ring3T sub-01:ringSE sub-04:ringSE; do $PYFSL $C/07_ds002702_gate_group_maps/kay_undistort.py ${x%:*} ${x#*:}; done
for x in sub-01:ring7T sub-03:ring7T sub-04:ring7T sub-01:ring3T sub-03:ring3T sub-01:ringSE sub-04:ringSE; do
  apptainer exec --bind $G --env $ENV,KAY_PREP=prep_dc $XCPD python3 $C/07_ds002702_gate_group_maps/kay_regcheck.py ${x%:*} ${x#*:}; done > $K/regcheck_stage2.log
$PY $C/07_ds002702_gate_group_maps/kay_stage2_select.py $K/regcheck_stage1.log $K/regcheck_stage2.log $K/stage2_selection.json

step "within-session emulation (7T 0.8 mm; x1.3125, x2, x3), summary, pre-specification for the cross-session test"
for S in sub-01 sub-02 sub-03 sub-04 sub-05; do for f in 1.3125 2 3; do $PY $C/07_ds002702_gate_group_maps/kay_emul.py $S --factor $f; done; done
$PY $C/07_ds002702_gate_group_maps/summarize_kay.py $K/emul $K/summary
$PY $C/07_ds002702_gate_group_maps/kay_prespec.py $K/emul $K/summary/kay_prespec.json     # before any cross-session score is read

step "real second acquisitions: 3T 2.4 mm and 7T SE 1.05 mm -> 7T GE 0.8 mm, uncorrected and field-map corrected"
for x in sub-01:ring3T sub-03:ring3T sub-01:ringSE sub-04:ringSE; do
  $PY $C/07_ds002702_gate_group_maps/kay_cross.py ${x%:*} ${x#*:}; $PY $C/07_ds002702_gate_group_maps/kay_cross.py ${x%:*} ${x#*:} --dc; done
$PY $C/07_ds002702_gate_group_maps/summarize_kay_cross.py $K/cross $K/summary/kay_prespec.json $K/summary
$PY $C/07_ds002702_gate_group_maps/kay_cross_cv.py $K/cross $K/summary                     # second analysis: settings chosen on the other subject

step "weeks 8-9: contamination-weighted leak (raw; denoised)"
SHA8=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN.sha1" | head -1); SHA9=$(cut -d' ' -f1 "$HERE/plans/STATS_PLAN_w9.sha1" | head -1)
for S in sub-01 sub-02 sub-03 sub-04 sub-05; do for f in 1.3125 2 3; do $PY $C/08_contamination/cwl_kay.py $S --factor $f --plan-sha1 $SHA8; done; done
$PY $C/08_contamination/summarize_cwl.py kay $K/emul $K/cwl $K/summary_w8/kay_cwl_summary.json
python3 - "$K/summary_w8/kay_cwl_summary.json" > /tmp/kay_cwl2_jobs.txt <<'EOF'
import json, sys
for fac, c in json.load(open(sys.argv[1]))['cells'].items():
    for S, r in c.items():
        print(S, fac, r['model'])
EOF
while read S f m; do $PY $C/09_contamination_denoised/cwl2_kay.py $S --factor $f --model $m --plan-sha1 $SHA9; done < /tmp/kay_cwl2_jobs.txt
$PY $C/09_contamination_denoised/summarize_cwl2.py kay $K/emul $K/cwl2 $K/summary_w9/kay_cwl2_summary.json
echo "ds002702 done; the tests and figures are built by run_paper.sh"
