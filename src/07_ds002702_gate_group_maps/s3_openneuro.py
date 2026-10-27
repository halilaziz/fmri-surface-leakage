"""List or download an OpenNeuro dataset from the public S3 bucket (standard library only; no installs).
Usage: python3 s3_openneuro.py list DATASET OUTJSON
       python3 s3_openneuro.py get DATASET LISTJSON DESTDIR [--jobs 8] [--prefix sub-01/]
Downloads to DEST/<key without the dataset prefix>, writing *.part then renaming; skips files already present with the right size.
"""
import sys, os, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BASE = 'https://s3.amazonaws.com/openneuro.org'
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def list_keys(ds):
    keys, token = [], None
    while True:
        q = {'list-type': '2', 'prefix': f'{ds}/', 'max-keys': '1000'}
        if token:
            q['continuation-token'] = token
        with urllib.request.urlopen(f'{BASE}?{urllib.parse.urlencode(q)}', timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall('s3:Contents', NS):
            keys.append((c.find('s3:Key', NS).text, int(c.find('s3:Size', NS).text)))
        if root.find('s3:IsTruncated', NS).text != 'true':
            return keys
        token = root.find('s3:NextContinuationToken', NS).text


def get_one(key, size, ds, dest):
    out = os.path.join(dest, key[len(ds) + 1:])
    if os.path.exists(out) and os.path.getsize(out) == size:
        return 'skip', size
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f'{BASE}/{urllib.parse.quote(key)}', timeout=120) as r, open(out + '.part', 'wb') as f:
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(out + '.part') == size:
                os.replace(out + '.part', out); return 'ok', size
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return 'fail', size


def main():
    mode, ds = sys.argv[1], sys.argv[2]
    if mode == 'list':
        keys = list_keys(ds)
        json.dump(keys, open(sys.argv[3], 'w'))
        print(len(keys), 'files', round(sum(s for _, s in keys) / 1e9, 2), 'GB')
        return
    keys = json.load(open(sys.argv[3])); dest = sys.argv[4]
    jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8
    if '--prefix' in sys.argv:
        pre = sys.argv[sys.argv.index('--prefix') + 1]; keys = [k for k in keys if k[0][len(ds) + 1:].startswith(pre)]
    t0 = time.time(); done = 0; tot = sum(s for _, s in keys); stat = {'ok': 0, 'skip': 0, 'fail': 0}
    with ThreadPoolExecutor(jobs) as ex:
        for st, s in ex.map(lambda kv: get_one(kv[0], kv[1], ds, dest), keys):
            stat[st] += 1; done += s
            if sum(stat.values()) % 50 == 0 or sum(stat.values()) == len(keys):
                el = time.time() - t0
                print(f'{sum(stat.values())}/{len(keys)} files, {done / 1e9:.1f}/{tot / 1e9:.1f} GB, {done / 1e6 / max(el, 1):.1f} MB/s, {stat}', flush=True)
    print('finished', stat, flush=True)
    fails = [k for k, s in keys if not (os.path.exists(os.path.join(dest, k[len(ds) + 1:])) and os.path.getsize(os.path.join(dest, k[len(ds) + 1:])) == s)]
    json.dump(fails, open(os.path.join(dest, '..', f'{ds}_missing.json'), 'w'))


if __name__ == '__main__':
    main()
