#!/bin/bash
# NYU Retinotopy Dataset (OpenNeuro ds003787, 3T, 2.0 mm, sweeping bars): external replication (plans/PLAN_NYU.md, its deviations
# plans/DEVIATIONS_NYU.md) and the three further analyses of plans/PLAN_NYU_GATE.md. Every step in order, with its command.
# Runs on a server with Apptainer: FSL / FreeSurfer steps inside the fMRIPrep container, Python + Workbench steps inside the XCP-D
# container (see ENVIRONMENT.md). Workspace: GEO_KAY_ROOT (data/ds003787,
# results/nyu/...). The summaries run locally on the copied score files (python3 with numpy, scipy, matplotlib).
set -euo pipefail
: "${GEO_KAY_ROOT:?set GEO_KAY_ROOT to the workspace}"
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/src; W=$GEO_KAY_ROOT
FMRIPREP=${FMRIPREP:-$W/code/run_fmriprep.sh}; XCPD=${XCPD:-$W/code/run_xcpd.sh}
cd "$W"
# 1. download (public S3 bucket; the 44 subjects with FreeSurfer reconstructions, bar-sweep runs, field maps, stimulus apertures)
python3 $C/07_ds002702_gate_group_maps/s3_openneuro.py get ds003787 $HERE/subjects/ds003787_subset_keys.json data/ds003787 --jobs 8
# 2. split: plans/nyu_split.json (seed 17; 8 tuning, 36 test subjects); the summaries read it from GEO_PROJECT/results/week17
export GEO_PROJECT=$W; mkdir -p results/week17 && cp $HERE/plans/nyu_split.json results/week17/
TUNE=$(python3 -c "import json; print(' '.join(json.load(open('$HERE/plans/nyu_split.json'))['tuning']))")
TEST=$(python3 -c "import json; print(' '.join(json.load(open('$HERE/plans/nyu_split.json'))['test']))")
# 3. preprocessing per subject (topup, motion correction to the single-band reference, applytopup, BBR; ~19 min each)
for S in $TUNE $TEST; do $FMRIPREP python3 $C/17_nyu_3t_replication/nyu_prep.py $S --threads 6; done
# 4. tuning subjects: full prior grid at both coarsenings (4.5-6 h each); choose the settings (writes results/week17/SETTINGS_NYU.txt)
for S in $TUNE; do $XCPD python3 $C/17_nyu_3t_replication/nyu_emul.py $S; done
python3 $C/17_nyu_3t_replication/nyu_summary.py tune results/nyu/emul results/week17/nyu_tuning
# 5. test subjects at the chosen settings (28-77 min each), then the pre-specified tests N1-N3 and Fig. 10
SET=$(cat results/week17/SETTINGS_NYU.txt)
for S in $TEST; do $XCPD python3 $C/17_nyu_3t_replication/nyu_emul.py $S --settings "$SET" --save-maps; done
python3 $C/17_nyu_3t_replication/nyu_summary.py test results/nyu/emul results/week17/nyu_tuning.json results/week17/nyu_test
python3 $C/17_nyu_3t_replication/fig_week17.py results/week17/figW17-1_nyu_replication.png --test results/week17/nyu_test.json
# 6. PLAN_NYU_GATE.md: every included subject again at the chosen settings, with CWL-P, the gated estimator and field statistics
for S in $TUNE $TEST; do $XCPD python3 $C/17_nyu_3t_replication/nyu_emul.py $S --settings "$SET" --gate --save-maps --out results/nyu/emul_gate; done
python3 $C/17_nyu_3t_replication/nyu_gate_summary.py results/nyu/emul_gate          # A (G1/G2) and B (registration) -> results/week17/nyu_gate.md
# 7. PLAN_ATLAS.md (week 18): contamination from the anatomical atlas of Benson et al. (2014), version 4.0. Take the template files
#    ?h.benson14_{angle,eccen,varea}.v4_0.mgz from the neuropythy 0.13.0 source archive on PyPI
#    (neuropythy-0.13.0/src/neuropythy/lib/data/fsaverage/surf/) into data/atlases/benson14_v4_0/surf/; no neuropythy code is needed.
$FMRIPREP python3 $C/18_atlas_contamination/atlas_map.py $TEST                      # results/nyu/atlas/S.benson14.npz
for S in $TEST; do $XCPD python3 $C/17_nyu_3t_replication/nyu_emul.py $S --settings "$SET" --gate --atlas --out results/nyu/emul_atlas; done
mkdir -p results/week18 && cp $HERE/plans/PLAN_ATLAS.* results/week18/
python3 $C/18_atlas_contamination/atlas_summary.py results/nyu/emul_atlas           # tests A1/A2 -> results/week18/atlas_test.md
# 8. 3T group leakage maps on fsaverage (C) and their figure (Figure S2)
$FMRIPREP python3 $C/17_nyu_3t_replication/nyu_groupmaps.py $W/results/week17/groupmaps
python3 $C/17_nyu_3t_replication/fig_groupmaps3t.py results/week17/groupmaps
echo "results: results/week17/{nyu_tuning,nyu_test,nyu_gate}.md, results/week17/groupmaps/, results/week18/atlas_test.md"
