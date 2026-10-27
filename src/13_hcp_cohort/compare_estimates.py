"""Week 13 regression check: the generalised week-3 fit (stimulus / segment options, defaults = wedge, full) must reproduce the
stored pilot estimates. Compares every array of two estimate files; the solver is iterative, so small differences are allowed.
Usage: python3 compare_estimates.py OLD.npz NEW.npz [rtol] [allow-missing-regex]
The raw posterior fields (f_*, sim_f_*) come from a PCG solve with tolerance 1e-4, so they are allowed 10 x rtol.
"""
import sys
import numpy as np

import re
old, new = np.load(sys.argv[1]), np.load(sys.argv[2]); rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
allow = re.compile(sys.argv[4]) if len(sys.argv) > 4 else None
missing = sorted(set(old.files) - set(new.files)); extra = sorted(set(new.files) - set(old.files))
missing_bad = [k for k in missing if not (allow and allow.search(k))]
worst, bad = 0.0, []
for k in sorted(set(old.files) & set(new.files)):
    a, b = old[k], new[k]
    if a.shape != b.shape:
        bad.append(f'{k}: shape {a.shape} vs {b.shape}'); continue
    if a.dtype.kind in 'iub' or b.dtype.kind in 'iub':
        if not np.array_equal(a, b):
            bad.append(f'{k}: integer/boolean arrays differ'); continue
        continue
    num = np.linalg.norm((a - b).ravel()); den = max(np.linalg.norm(a.ravel()), 1e-30)
    rel = float(num / den); worst = max(worst, rel)
    tol = 10 * rtol if (k.startswith('f_') or k.startswith('sim_f_')) else rtol
    if rel > tol:
        bad.append(f'{k}: relative difference {rel:.2e} (tolerance {tol:.0e})')
print(f'{len(old.files)} arrays; worst relative difference {worst:.2e}; missing in new: {missing}; extra in new: {extra}')
if missing_bad:
    bad.append('arrays missing in new: ' + ', '.join(missing_bad))
if bad:
    print('DIFFERENCES:'); [print('  ' + x) for x in bad]; sys.exit(1)
print('PASS: identical within rtol', rtol)
