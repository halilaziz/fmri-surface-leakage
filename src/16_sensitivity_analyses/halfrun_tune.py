"""Week 16, E1: re-tune the model and the competitors on half-run sources (5 tuning subjects).
subject mode (xcp_d container): score every method of one half-run estimate file (week-3 strata; the references are the full
runs' 1.6 mm maps, as stored by run_subject.py --segment). summary mode (local): average the two halves per subject and the
subjects, pick the settings by the week-3 rule (lowest mean residual over all coherent vertices; also where leak > 10%).
Usage: python3 halfrun_tune.py subject S HALF_DIR OUT_DIR      -> OUT_DIR/S.HALF.json  (HALF = basename of HALF_DIR)
       python3 halfrun_tune.py summary IN_DIR OUT_PREFIX
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_hcp_emulation'))


def subject(S, half_dir, out):
    from evaluate import Merged, subject_context, score_subject
    E = Merged([f'{half_dir}/{S}/{S}.week3_estimates.npz'])
    ctx = subject_context(S, E['idx'])
    sc = score_subject(E, ctx)
    os.makedirs(out, exist_ok=True)
    json.dump({'subject': S, 'half': os.path.basename(half_dir), 'scores': sc}, open(f'{out}/{S}.{os.path.basename(half_dir)}.json', 'w'), indent=1, default=float)
    print(S, os.path.basename(half_dir), {g: sorted(((m, round(v['coherent'], 4)) for m, v in sc[g].items() if m != '_n'), key=lambda t: t[1])[:3] for g in sc}, flush=True)


def summary(indir, prefix):
    J = [json.load(open(f)) for f in sorted(glob.glob(f'{indir}/*.json'))]
    subs = sorted({j['subject'] for j in J}); halves = sorted({j['half'] for j in J})
    full = json.load(open('results/week3/week3_tuning.json'))['choice']
    res = {'subjects': subs, 'halves': halves, 'choice': {}, 'table': {}}
    L = [f'# Week 16 E1: settings re-tuned on half-run sources ({len(subs)} tuning subjects, halves {halves})', '',
         'Cross-run residual against the other run\'s full-run 1.6 mm map, averaged over the two halves and the subjects; the week-3 rule picks the',
         'lowest mean over all coherent vertices (first table) and, secondarily, where leak > 10% at the coarse grid. Full-run choices from week 3 for comparison.', '']
    for g in ('2.4', '3.2'):
        names = sorted({m for j in J for m in j['scores'].get(g, {}) if m != '_n'})
        avg = {}
        for crit in ('coherent', 'clean16_leak10'):
            avg[crit] = {}
            for m in names:
                per = []
                for S in subs:
                    v = [j['scores'][g][m][crit] for j in J if j['subject'] == S and g in j['scores'] and m in j['scores'][g] and j['scores'][g][m].get(crit) is not None]
                    if v:
                        per.append(np.mean(v))
                if per:
                    avg[crit][m] = (float(np.mean(per)), len(per))
        pick = lambda crit, pfx: min((m for m in avg[crit] if m.startswith(pfx)), key=lambda m: avg[crit][m][0])
        ch = dict(model=pick('coherent', 'model_'), smooth=pick('coherent', 'smooth'), upsmooth=pick('coherent', 'upsmooth'),
                  model_leak=pick('clean16_leak10', 'model_'), smooth_leak=pick('clean16_leak10', 'smooth'), upsmooth_leak=pick('clean16_leak10', 'upsmooth'))
        res['choice'][g] = ch; res['table'][g] = avg
        L += [f'## {g} mm', '', f"Half-run choice (all coherent): model **{ch['model']}**, smoothing **{ch['smooth']}**, upsampling + smoothing **{ch['upsmooth']}**; "
              f"full-run choice (week 3): {full[g]['model']}, {full[g]['smooth']}, {full[g]['upsmooth']}. Where leak > 10%: {ch['model_leak']}, {ch['smooth_leak']}, {ch['upsmooth_leak']}.", '',
              '| method | all coherent | leak > 10% |', '|---|---|---|']
        for m in sorted(names, key=lambda m: avg['coherent'].get(m, (9, 0))[0]):
            if m.startswith('model_') or m.startswith('smooth') or m.startswith('upsmooth') or m in ('standard', 'upsample'):
                L.append(f"| {m} | {avg['coherent'][m][0]:.4f} | {avg['clean16_leak10'].get(m, (float('nan'), 0))[0]:.4f} |")
        L.append('')
    open(f'{prefix}.md', 'w').write('\n'.join(L) + '\n'); json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    print('\n'.join(L[:12]))


if __name__ == '__main__':
    if sys.argv[1] == 'subject':
        subject(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        summary(sys.argv[2], sys.argv[3])
