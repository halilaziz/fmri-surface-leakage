"""Week 7, ds002702 stage 2: per session, use the field-map corrected data (prep_dc) only if the correction improved the
registration QC of kay_regcheck.py (fraction of white-surface vertices with grey > white EPI intensity), else the uncorrected
data (prep). Usage: python3 kay_stage2_select.py REGCHECK_STAGE1_LOG REGCHECK_STAGE2_LOG OUT_JSON
"""
import sys, json, ast


def frac(path):
    out = {}
    for line in open(path):
        if 'registered' in line:
            S, ses = line.split()[:2]; out[f'{S}/{ses}'] = ast.literal_eval(line[line.index('{'):].strip())['frac_positive']
    return out


f1, f2 = frac(sys.argv[1]), frac(sys.argv[2])
sel = {k: dict(use='prep_dc' if f2[k] > f1[k] else 'prep', qc_uncorrected=f1[k], qc_corrected=f2[k]) for k in f2 if k in f1}
json.dump(sel, open(sys.argv[3], 'w'), indent=1)
print(json.dumps(sel, indent=1))
