# Cortical-sheet sampling theory: numerical checks

Reproduces Appendix A of ../BLUEPRINT.md (~20 min on a laptop; numpy/scipy/matplotlib/nibabel/nilearn).

    python run_experiments.py      # T1/T2: orientation law, encodings, jitter, BOLD PSF, voxel size  -> results/results_T1_T2.json
    python run_banks_orient.py     # T3 (white prior) + T4 fsaverage5 orientation distribution         -> results/results_T3_T4.json
    python run_banks_realistic.py  # T3b: two-bank separability, BOLD-limited prior, CNR 0.5/2/10      -> results/results_T3b.json
    python make_figures.py         # figures/figA1..A4, results/table_two_bank.md

Model-internal only (planar patches, fsaverage5 average surface); see Appendix A caveats.
