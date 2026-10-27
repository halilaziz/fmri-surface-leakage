"""Week 4, StudyForrest: robust re-registration of every preprocessed session (fmriprep container).
The first pass initialised bbregister from FLIRT (correlation ratio); for some 7T sessions that start was poor and BBR ended in a
wrong optimum (cost ~0.95). Here bbregister is run from several starts and the lowest final cost is kept:
  flirt (first pass), coreg (FreeSurfer mri_coreg, NMI), header (scanner coordinates), and for the 0.8 mm slab also the best
  registration of the same session's large-coverage run (shared scanner space). Betas are unaffected (native space).
Usage: python sf_reregister.py SUBJECT [SESSION,SESSION] [--full]   (writes {S}.sf_rereg_qc.json; the prep QC file is left alone)
Default (quick): first-pass costs + anchoring only; --full also tries the coreg / header / coverage starts first.
"""
import os, sys, json, glob, shutil, subprocess
import numpy as np

ROOT = os.environ.get('GEO_ROOT', '/path/to/geo_root')
SF = f'{ROOT}/data/studyforrest'
PREP = f'{ROOT}/results/week4/sf/prep'
os.environ['SUBJECTS_DIR'] = f'{SF}/freesurfer'


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout[-1500:] + r.stderr[-800:]


def bbr(S, mov, out, init):
    """init: '--init-coreg', '--init-header' or '--init-reg FILE'. Returns final cost (inf on failure)."""
    for f in glob.glob(out.replace('.lta', '') + '*'):
        os.remove(f)
    code, msg = sh(f'bbregister --mov {mov} {init} --s {S} --bold --6 --reg {out}')
    mc = glob.glob(out.replace('.lta', '') + '*.mincost')
    return float(open(mc[0]).read().split()[0]) if code == 0 and mc else float('inf')


def best_of(S, mov, wd, tag, starts):
    """starts: dict name -> init string. Keeps the lowest-cost result as {wd}/{tag}.lta; returns (name, cost, all costs)."""
    costs = {}
    for name, init in starts.items():
        out = f'{wd}/cand_{tag}_{name}.lta'
        costs[name] = bbr(S, mov, out, init)
    name = min(costs, key=costs.get)
    shutil.copy(f'{wd}/cand_{tag}_{name}.lta', f'{wd}/{tag}.lta')
    return name, costs[name], costs


def main():
    args = [x for x in sys.argv[1:] if not x.startswith('--')]; full = '--full' in sys.argv
    S = args[0]; only = set(args[1].split(',')) if len(args) > 1 else None
    fs = f'{SF}/freesurfer/{S}'
    if not os.path.exists(f'{fs}/mri/brainmask.mgz'):
        os.symlink('brain.mgz', f'{fs}/mri/brainmask.mgz')      # mri_coreg's reference; brain.mgz is the skull-stripped T1
    qc_path = f'{PREP}/{S}/{S}.sf_rereg_qc.json'
    rq = json.load(open(qc_path)) if os.path.exists(qc_path) else {}
    if not full:
        # quick mode: first-pass (FLIRT-initialised) costs from the prep QC; only the anchoring pass below re-runs bbregister
        pq = json.load(open(f'{PREP}/{S}/{S}.sf_prep_qc.json'))['sessions']
        for key, v in pq.items():
            if 'bbr_cost' in v and key not in rq:
                rq[key] = dict(best='flirt', cost=v['bbr_cost'], costs={'flirt': v['bbr_cost']})
                shutil.copy(f'{PREP}/{S}/{key}/bbr.lta', f'{PREP}/{S}/{key}/bbr_flirt.lta')
    for key in (sorted(os.listdir(f'{PREP}/{S}')) if full else []):
        wd = f'{PREP}/{S}/{key}'
        if not os.path.exists(f'{wd}/betaR_even.nii.gz') or (only and key not in only):
            continue
        mean = f'{wd}/session_mean.nii.gz'
        if not os.path.exists(f'{wd}/bbr_flirt.lta') and os.path.exists(f'{wd}/bbr.lta'):
            shutil.copy(f'{wd}/bbr.lta', f'{wd}/bbr_flirt.lta')
        starts = {'coreg': '--init-coreg', 'header': '--init-header'}
        if os.path.exists(f'{wd}/bbr_flirt.lta'):
            starts['flirt'] = f'--init-reg {wd}/bbr_flirt.lta'
        if key == '7T-r08' and os.path.exists(f'{wd}/coverage/cov_mean.nii.gz'):
            cw = f'{wd}/coverage'
            cstarts = {'coreg': '--init-coreg', 'header': '--init-header'}
            if os.path.exists(f'{cw}/bbr.lta'):
                shutil.copy(f'{cw}/bbr.lta', f'{cw}/bbr_flirt_cov.lta'); cstarts['flirt'] = f'--init-reg {cw}/bbr_flirt_cov.lta'
            cname, ccost, ccosts = best_of(S, f'{cw}/cov_mean.nii.gz', cw, 'bbr_best', cstarts)
            code, _ = sh(f'lta_convert --inlta {cw}/bbr_best.lta --src {mean} --trg {fs}/mri/orig.mgz --outlta {wd}/init_from_cov.lta --subject {S}')
            if code == 0:
                starts['coverage'] = f'--init-reg {wd}/init_from_cov.lta'
            rq[f'{key}/coverage'] = dict(best=cname, cost=ccost, costs=ccosts)
        name, cost, costs = best_of(S, mean, wd, 'bbr', starts)
        rq[key] = dict(best=name, cost=cost, costs=costs)
        json.dump(rq, open(qc_path, 'w'), indent=1)
        print(S, key, 'best start', name, 'cost', round(cost, 3), {k: round(v, 3) for k, v in costs.items()}, flush=True)
    # pass 2: within-subject anchoring. The session with the lowest cost (usually 3 mm: largest slab) is the anchor; every session
    # still above 0.6 is aligned to the anchor's mean EPI (same scanner and contrast: FLIRT corratio, +-20 deg search), composed
    # with the anchor's BBR and refined with bbregister; kept if its cost is lower.
    keys = [k for k in rq if '/' not in k]
    if not keys:
        return
    anchor = min(keys, key=lambda k: rq[k]['cost'])
    aw = f'{PREP}/{S}/{anchor}'
    if rq[anchor]['cost'] > 0.65:
        print(S, 'no reliable anchor (best', anchor, round(rq[anchor]['cost'], 3), ')', flush=True); return
    rq['_anchor'] = dict(session=anchor, cost=rq[anchor]['cost'])

    def anchor_to(key, anc, tag):
        wd, aw = f'{PREP}/{S}/{key}', f'{PREP}/{S}/{anc}'; mean = f'{wd}/session_mean.nii.gz'
        code1, _ = sh(f'flirt -in {mean} -ref {aw}/session_mean.nii.gz -dof 6 -cost corratio -searchrx -20 20 -searchry -20 20 '
                      f'-searchrz -20 20 -omat {wd}/to_{tag}.mat')
        code2, _ = sh(f'lta_convert --infsl {wd}/to_{tag}.mat --src {mean} --trg {aw}/session_mean.nii.gz --outlta {wd}/to_{tag}.lta --subject {S}')
        code3, _ = sh(f'mri_concatenate_lta {wd}/to_{tag}.lta {aw}/bbr.lta {wd}/init_{tag}.lta')
        if code1 or code2 or code3:
            print(S, key, tag, 'failed', code1, code2, code3, flush=True); return
        c = bbr(S, mean, f'{wd}/cand_bbr_{tag}.lta', f'--init-reg {wd}/init_{tag}.lta')
        rq[key]['costs'][tag] = c
        if c < rq[key]['cost']:
            shutil.copy(f'{wd}/cand_bbr_{tag}.lta', f'{wd}/bbr.lta'); rq[key].update(best=tag, cost=c)
        json.dump(rq, open(qc_path, 'w'), indent=1)
        print(S, key, 'anchored to', anc, f'({tag})', 'cost', round(c, 3), '-> best', rq[key]['best'], round(rq[key]['cost'], 3), flush=True)

    for key in keys:
        if key == anchor or rq[key]['cost'] <= 0.6 or (only and key not in only) or 'anchor' in rq[key]['costs']:
            continue
        anchor_to(key, anchor, 'anchor')
    # pass 3: sessions still above 0.6 try the best-registered session of the same scanner (same contrast and distortions)
    for key in keys:
        if rq[key]['cost'] <= 0.6 or (only and key not in only) or 'anchor_same_field' in rq[key]['costs']:
            continue
        same = [k for k in keys if k != key and k.split('-')[0] == key.split('-')[0] and rq[k]['cost'] <= 0.5]
        if same:
            anc = min(same, key=lambda k: rq[k]['cost'])
            if anc != anchor:
                anchor_to(key, anc, 'anchor_same_field')


if __name__ == '__main__':
    main()
