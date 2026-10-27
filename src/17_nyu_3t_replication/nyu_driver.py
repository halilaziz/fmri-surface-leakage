"""Week 17 queue on hebb (host python3, standard library only; everything inside GEO_CONST).
Stage 1 (prep): as soon as all of a subject's selected files are downloaded with the right size, run nyu_prep.py in the fMRIPrep
container (tuning subjects first), at most MAXP at a time.
Stage 2 (emul): tuning subjects with the full settings grid as soon as their preprocessing is done; test subjects only after
results/nyu/SETTINGS_NYU.txt exists (written after the selection on the tuning subjects), with the chosen settings and maps
saved; at most MAXE at a time. Resumable: a stage is skipped when its output exists.
Usage: nohup python3 code/src/17_nyu_3t_replication/nyu_driver.py [--maxp 4] [--maxe 3] > logs/week17/driver.log &
"""
import os, sys, json, time, subprocess, argparse, collections

G = os.environ.get('GEO_KAY_ROOT', '/path/to/geo_kay_root')
CODE = f'{G}/code/src/17_nyu_3t_replication'
DATA = f'{G}/data/ds003787'
PREP = f'{G}/results/nyu/prep'; EMUL = f'{G}/results/nyu/emul'; LOG = f'{G}/logs/week17'


def expected_files():
    keys = json.load(open(f'{G}/data/ds003787_subset_keys.json')); per = collections.defaultdict(list)
    for k, s in keys:
        rel = k[len('ds003787/'):]
        for pre in ('', 'derivatives/freesurfer/', 'derivatives/stimulus_apertures/'):
            if rel.startswith(pre + 'sub-'):
                per[rel[len(pre):].split('/')[0]].append((rel, s))
    return per


def is_running(script, S):
    """True if a process of `script` for subject S already runs (e.g. started by an earlier driver instance)."""
    r = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', f'{script} {S}( |$)'], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ''


def complete(files):
    return all(os.path.exists(f'{DATA}/{r}') and os.path.getsize(f'{DATA}/{r}') == s for r, s in files)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--maxp', type=int, default=4); ap.add_argument('--maxe', type=int, default=3)
    a = ap.parse_args()
    split = json.load(open(f'{G}/results/nyu/nyu_split.json')); order = split['tuning'] + split['test']; tune = set(split['tuning'])
    per = expected_files(); running = {}
    os.makedirs(LOG, exist_ok=True); os.makedirs(EMUL, exist_ok=True)
    say = lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6')
    while True:
        for k, p in list(running.items()):
            if p.poll() is not None:
                say('finished', k, 'exit', p.returncode); del running[k]
        nprep = sum(1 for k in running if k[0] == 'prep'); nemul = sum(1 for k in running if k[0] == 'emul')
        settings = open(f'{G}/results/nyu/SETTINGS_NYU.txt').read().strip() if os.path.exists(f'{G}/results/nyu/SETTINGS_NYU.txt') else None
        pending = 0
        for S in order:
            qcf = f'{PREP}/{S}/{S}_prep_qc.json'; scf = f'{EMUL}/{S}.nyu_scores.json'
            done_emul = os.path.exists(scf) and 't_total_s' in json.load(open(scf))
            if done_emul or (os.path.exists(f'{LOG}/emul_{S}.failed')):
                continue
            pending += 1
            if ('prep', S) in running or ('emul', S) in running or is_running('nyu_prep.py', S) or is_running('nyu_emul.py', S):
                continue
            if not os.path.exists(qcf):
                if os.path.exists(f'{LOG}/prep_{S}.failed') or nprep >= a.maxp or not complete(per[S]):
                    continue
                cmd = f'{G}/code/run_fmriprep.sh python3 {CODE}/nyu_prep.py {S} --threads 4'
                running[('prep', S)] = subprocess.Popen(f'nice -n 10 {cmd} > {LOG}/prep_{S}.log 2>&1 || touch {LOG}/prep_{S}.failed', shell=True, env=env)
                nprep += 1; say('prep started', S); continue
            if nemul >= a.maxe:
                continue
            if S in tune:
                extra = ''
            elif settings:
                extra = f"--settings '{settings}' --save-maps"
            else:
                continue
            cmd = f'{G}/code/run_xcpd.sh python3 {CODE}/nyu_emul.py {S} {extra}'
            running[('emul', S)] = subprocess.Popen(f'nice -n 10 {cmd} > {LOG}/emul_{S}.log 2>&1 || touch {LOG}/emul_{S}.failed', shell=True, env=env)
            nemul += 1; say('emul started', S, '(tuning)' if S in tune else '(test)')
        if pending == 0 and not running:
            say('all subjects done'); break
        time.sleep(60)


if __name__ == '__main__':
    main()
