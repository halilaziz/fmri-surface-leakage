"""Week 14: verify the subject sets against the HCP restricted family table and record hashes of the lists.

Checks (all must hold): the 5 tuning and the 15 development subjects belong to 20 different families; no confirmatory-cohort
subject shares a family with a tuning subject; the cohort and the development set are disjoint; the sensitivity set consists of
relatives of tuning subjects; the primary set is the lowest subject ID of each cohort family. Writes counts and SHA-256 sums of
the sorted ID lists (tuning, development, cohort, sensitivity, primary subset) — no IDs of the primary subset or the sensitivity
set are written, since those encode family structure. Needs the restricted table (local only).
The tuning and development sets are read from the study-code key (code, hcp_id, role), which is not published: the key is
submitted to the HCP on acceptance (Restricted Data Use Terms), so HCP restricted-data users can obtain it there.
Usage (local): python3 src/14_robustness_checks/verify_splits.py OUT_MD [--family-table configs/splits_7T_family.tsv] [--batches results/week13/batches]
               [--key configs/hcp_study_codes.tsv]
"""
import csv, json, glob, hashlib, argparse

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--family-table', default='configs/splits_7T_family.tsv')
ap.add_argument('--batches', default='results/week13/batches'); ap.add_argument('--release-lists', default='release/subjects')
ap.add_argument('--key', default='configs/hcp_study_codes.tsv')
a = ap.parse_args()
fam = {r['Subject']: r['Family_ID'] for r in csv.DictReader(open(a.family_table), delimiter='\t')}
K = list(csv.DictReader(open(a.key), delimiter='\t'))
TUNE = sorted(r['hcp_id'] for r in K if r['role'] == 'tuning')
DEV = sorted(r['hcp_id'] for r in K if r['role'] == 'test')
cohort = sorted({s.strip() for b in sorted(glob.glob(f'{a.batches}/batch_*.txt')) if 'batch_17' not in b for s in open(b) if s.strip()})
sens = sorted(s.strip() for s in open(f'{a.batches}/batch_17.txt') if s.strip())
keep = {}
for S in cohort:
    keep[fam[S]] = min(keep.get(fam[S], S), S)
primary = sorted(keep.values())
tune_f = {fam[s] for s in TUNE}; dev_f = {fam[s] for s in DEV}
checks = {
    'tuning and development subjects in 20 different families': len(tune_f | dev_f) == 20 and len(tune_f) == 5 and len(dev_f) == 15,
    'no cohort subject in a tuning family': not any(fam[s] in tune_f for s in cohort),
    'cohort and development set disjoint': not set(cohort) & set(DEV) and not set(cohort) & set(TUNE),
    'sensitivity subjects are relatives of tuning subjects': all(fam[s] in tune_f for s in sens) and not set(sens) & set(TUNE),
    'cohort size 158 in 86 families': len(cohort) == 158 and len(keep) == 86,
    'primary set = lowest ID per family (86)': len(primary) == 86 and all(min(x for x in cohort if fam[x] == fam[p]) == p for p in primary),
}
h = lambda lst: hashlib.sha256(('\n'.join(sorted(lst)) + '\n').encode()).hexdigest()
L = ['# Subject sets: checks against the HCP restricted family table, and hashes of the lists', '',
     'Produced by `src/14_robustness_checks/verify_splits.py` (needs the restricted table, which is not distributed). Hashes are SHA-256 of the sorted',
     'subject IDs, one per line with a trailing newline; a user with HCP restricted access can regenerate each list and compare.', '',
     '| check | result |', '|---|---|'] + [f'| {k} | {"pass" if v else "FAIL"} |' for k, v in checks.items()] + ['',
     '| list | n | families | SHA-256 |', '|---|---|---|---|',
     f'| tuning set | {len(TUNE)} | {len(tune_f)} | {h(TUNE)} |', f'| development set | {len(DEV)} | {len(dev_f)} | {h(DEV)} |',
     f'| confirmatory cohort | {len(cohort)} | {len(keep)} | {h(cohort)} |', f'| sensitivity set | {len(sens)} | {len({fam[s] for s in sens})} | {h(sens)} |',
     f'| primary subset (one per family) | {len(primary)} | {len(primary)} | {h(primary)} |']
open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
assert all(checks.values()), checks
