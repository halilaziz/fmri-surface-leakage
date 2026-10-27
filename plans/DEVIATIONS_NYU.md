# Week 17 — deviations from PLAN_NYU.md (recorded before any test-subject result)

1. **Prior grid extended (2026-10-06, about 14:50 local; no test subject analysed).** In the first tuning subject
   (sub-wlsubj057, 3.0 mm, direction A→B) the model's residual fell steadily towards the planned grid's weakest prior and
   shortest range (best r = 6 mm, s = 0.5; e.g. s 0.0625 → 0.5 at r = 6: 0.633 → 0.528), so the optimum lay outside the
   planned grid r ∈ {6, 12, 24, 48} × s ∈ {0.0625, 0.125, 0.25, 0.5}. The grid was extended to r ∈ {3, 6, 12, 24, 48} ×
   s ∈ {0.0625, 0.125, 0.25, 0.5, 1, 2, 4} for all tuning subjects. The same extension was needed for ds002702 (week 7),
   whose FreeSurfer-based pipeline uses the same σ0 definition. The selection rule is unchanged.
2. **Implementation fixes in preprocessing (no change of definition).** (a) Three subjects (sub-wlsubj086, 087, 088) have no
   single-band reference; the motion-correction reference is then the mean of run 1's first 10 volumes. (b) Ten runs of one
   subject and one subject's field maps are named without the `acq-PA` / `run-01` entities; files are matched with or without
   them. (c) FSL tools that exit with status 0 after failing to read an image are caught by checking their output files.
