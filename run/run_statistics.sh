#!/bin/bash
# The paper's pre-specified tests, summary tables and result figures, from the result files of the four dataset pipelines.
# Runs locally (python3 with numpy, scipy, matplotlib). Expects the project layout of the original runs:
#   PROJECT/results/week{1..17}/...  (copy the summary JSONs and figures from GEO_ROOT/results and GEO_KAY_ROOT/results/kay,
#   including results/week2/paper_extract from run_hcp.sh and the NYU score files and group maps from run_nyu.sh).
# HCP_FAMILY_TABLE: the HCP restricted family table (not distributed); without it the cohort's one-per-family rule cannot run.
set -euo pipefail
: "${PROJECT:?set PROJECT to the folder holding results/}"
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$HERE/src
cd "$PROJECT"
R=results
python3 $C/08_contamination/stats_w8.py $R/week8/summary/hcp_cwl_summary.json $R/week8/summary/sf_cwl_summary.json $R/week8/summary/kay_cwl_summary.json \
        $R/week8/sf_cross/sf_cross_loso.json $R/week8/STATS_PLAN.sha1 $R/week8
python3 $C/09_contamination_denoised/stats_w9.py $R/week9/summary/hcp_cwl2_summary.json $R/week9/summary/sf_cwl2_summary.json $R/week9/summary/kay_cwl2_summary.json \
        $R/week8/summary/hcp_cwl_summary.json $R/week8/summary/sf_cwl_summary.json $R/week8/summary/kay_cwl_summary.json \
        $R/week9/STATS_PLAN_w9.sha1 $R/week9
python3 $C/07_ds002702_gate_group_maps/plot_kay_cross.py $R/week7/ds002702/summary/kay_cross_results.json $R/week7/ds002702/summary/kay_cross_cv_results.json $R/week7/figW7-3_kay_cross.png
mkdir -p $R/figures && python3 $C/09_contamination_denoised/fig1_concept.py $R/figures/fig1_concept.png
# the confirmatory cohort (per-batch evaluations under results/week13/cohort/<batch>/eval from the cluster
# cohort jobs); the family rule needs the HCP restricted family table (see README) — else --family-rule none
FAM=${HCP_FAMILY_TABLE:-configs/splits_7T_family.tsv}
python3 $C/13_hcp_cohort/cohort_stats.py $R/week13/cohort_final --family-rule lowest --family-table $FAM --label "confirmatory cohort: 158 subjects, primary on one subject per family"
python3 $C/13_hcp_cohort/primary_holm.py $R/week13                 # post hoc: Holm / Bonferroni across the eight primary tests (S13.7)
python3 $C/13_hcp_cohort/cohort_stats.py $R/week13/cohort_sensitivity --batches $(ls -d $R/week13/cohort/batch_* | xargs -n1 basename | paste -sd, -) --family-rule lowest --family-table $FAM --label "sensitivity: cohort + 5 relatives of tuning subjects"
python3 $C/13_hcp_cohort/cohort_stats.py $R/week13/sensitivity_alone --batches batch_17 --family-rule none --family-table $FAM --label "sensitivity set alone (5 subjects, descriptive)"
python3 $C/13_hcp_cohort/cohort_stats.py $R/week13/pilot_explore --batches pilot --family-rule none --family-table $FAM --label "pilot development set, 15 subjects (exploratory)"
python3 $C/13_hcp_cohort/fig_cohort.py $R/week13/figW13-1_cohort.png $R/week13/cohort_curves.json
python3 $C/11_summary_table/robustness_table.py $R/week11   # the cross-dataset table (needs $R/week11/TABLE_SPEC.sha1 and the ds002702 emul npz files)
python3 $C/14_robustness_checks/ceiling.py $R/week14/ceiling                     # HCP noise ceiling (from the week-3 evaluation files)
python3 $C/14_robustness_checks/dose_sensitivity.py $R/week14/dose_sensitivity --family-table $FAM   # bin conventions (cohort)
# far-part tests (per-subject files from the ablation, realistic-kernel and real-session jobs)
python3 $C/15_far_part_tests/ablation_eval.py summary $R/week15/ablation/eval $R/week15/ablation_summary
python3 $C/15_far_part_tests/leak_realistic_summary.py $R/week15/leak_realistic $R/week15/leak_realistic_summary
python3 $C/15_far_part_tests/sf_operator_test.py summary $R/week15/sf_operator $R/week15/sf_operator_summary
python3 $C/15_far_part_tests/fig_week15.py $R/week15/figW15-1_reviewer_tests.png
# sensitivity analyses (per-subject files from the kernel-width, half-run, correlated-noise and far-criterion jobs)
python3 $C/16_sensitivity_analyses/psf_components.py $R/week16/psf_components.md
python3 $C/16_sensitivity_analyses/summaries.py psf $R/week16/psf_scan/eval $R/week16/psf_scan_summary
python3 $C/16_sensitivity_analyses/summaries.py thr $R/week16/leak_thresholds $R/week16/leak_thresholds_summary
python3 $C/16_sensitivity_analyses/summaries.py geo $R/week16/geodesic $R/week16/geodesic_summary
python3 $C/16_sensitivity_analyses/summaries.py leakpsf $R/week16/leak_psf $R/week16/leak_psf_summary
python3 $C/16_sensitivity_analyses/summaries.py corr $R/week16/corr/eval $R/week16/corr_summary
python3 $C/16_sensitivity_analyses/summaries.py f2 $R/week16/corr_grid/eval $R/week16/corr/eval $R/week16/corr_retuned
python3 $C/16_sensitivity_analyses/halfrun_tune.py summary $R/week16/halfrun_tune/eval $R/week16/halfrun_tune_summary
python3 $C/16_sensitivity_analyses/summaries.py h3 $R/week16/corr/eval $R/week16/halfrun_dev/eval_half1 $R/week16/halfrun_dev/eval_half2 $R/week16/h3_retuned \
        'model_r12_s0.125|model_r24_s0.25' 'upsmooth8|upsmooth8' 'model_r24_s0.25|model_r48_s0.5' 'upsmooth8|upsmooth10'
python3 $C/16_sensitivity_analyses/fig_week16.py $R/week16/figW16-1_reviewer_tests.png
# NYU Retinotopy Dataset (per-subject score files and group maps from run/run_nyu.sh)
export GEO_PROJECT=$PROJECT
python3 $C/17_nyu_3t_replication/nyu_summary.py tune $R/week17/emul $R/week17/nyu_tuning
python3 $C/17_nyu_3t_replication/nyu_summary.py test $R/week17/emul $R/week17/nyu_tuning.json $R/week17/nyu_test
python3 $C/17_nyu_3t_replication/fig_week17.py $R/week17/figW17-1_nyu_replication.png --test $R/week17/nyu_test.json
python3 $C/17_nyu_3t_replication/nyu_gate_summary.py $R/week17/emul_gate
python3 $C/17_nyu_3t_replication/fig_week17.py $R/week17/figW17-3_nyu_v2.png --test $R/week17/nyu_test.json --gate $R/week17/nyu_gate.json --rows 2
python3 $C/17_nyu_3t_replication/fig_groupmaps3t.py $R/week17/groupmaps   # needs groupmaps/_draw written by nyu_groupmaps.py
python3 $C/18_atlas_contamination/atlas_summary.py $R/week18/emul_atlas   # contamination from the anatomical atlas (PLAN_ATLAS.md) -> results/week18/atlas_test.md
echo "tests: results/week8/w8_stats.md, results/week9/w9_stats.md, results/week13/cohort_final.md, results/week17/nyu_test.md, nyu_gate.md,"
echo "results/week18/atlas_test.md; tables and figures under results/weekN/"
