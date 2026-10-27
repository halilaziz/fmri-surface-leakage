# Subject sets: SHA-256 of the sorted HCP subject IDs (one per line, trailing newline)

A user with HCP restricted access can rebuild each set with `src/14_robustness_checks/verify_splits.py` and the study-code key
(deposited with the HCP) and compare. The two 5-subject sets (tuning, sensitivity) are not hashed here: with 184 candidate
subjects they could be recovered by enumeration, which would expose family links.

| list | n | families | SHA-256 |
|---|---|---|---|
| development set | 15 | 15 | 11d13b63c53e1409e26f6aa9b8d5530ef4aef9e1e97e33c0280c8f9e7b82069b |
| confirmatory cohort | 158 | 86 | 0818e0505807d325f585739500b4b26d6e32b4973b20fe4c5c86474ff8784afa |
| primary subset (one per family) | 86 | 86 | 6f0b3348dd5432114cf6f06fc9df6c0937a4952f6a92254a51cb99a81abbe114 |
