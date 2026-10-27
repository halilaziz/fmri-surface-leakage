*Release copy: the five subject IDs are withheld (HCP restricted family information); the SHA-1 in `COHORT_PLAN_AMENDMENT_1.sha1` is of the unredacted file.*

# Amendment 1 to the week-13 cohort plan — sensitivity set (2026-10-03)

Decided by the user after the interim look at 8 of 16 batches ("add the 6 as a sensitivity set and continue"). The hashed plan
(`COHORT_PLAN.md`, SHA-1 21822393…) is not changed; this file is hashed separately (`COHORT_PLAN_AMENDMENT_1.sha1`).

## The set

The six HCP 7T subjects excluded from the confirmatory cohort were [five subject IDs withheld in this copy: restricted family information] (relatives of
the five tuning subjects; one of them also lacks the MSMAll spheres, which only the fs_LR group maps need) and [HCP ID withheld] (no
concatenated retinotopy run in the HCP release; it cannot be processed by this pipeline). **The sensitivity set is therefore
the five processable subjects**, run through the identical pipeline (stage 0, A, B, group) as `batch_17`.

## How it is used

- It is **not** part of the primary or secondary tests of H1–H4, whose subject set stays as hashed (158 subjects, 86 families).
- Two additional, clearly labelled analyses:
  1. the five alone, descriptive (means and intervals; no tests with n = 5);
  2. a pooled sensitivity analysis of every row of the plan over cohort + sensitivity set (163 subjects), with the same family
     rule (lowest subject ID per family), to show whether excluding the tuning subjects' relatives changed any conclusion.
- Rationale for the original exclusion stands: these subjects share families with the five subjects on which the settings were
  chosen, so they are not independent of the tuning data. The sensitivity analysis quantifies how much that matters.

## Implementation

`work/batches/batch_17.txt` on TRUBA; fetched by a one-off; stage 0 queued by it, fits by the fits pump; scored by the same group
job. Locally, `src/week13/cohort_stats.py` excludes `batch_17` and `pilot` from the default batch list; the sensitivity analyses
are run by naming the batches explicitly.
