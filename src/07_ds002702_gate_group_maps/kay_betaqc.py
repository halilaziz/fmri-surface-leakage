"""QC of ds002702 per-run betas: per run, voxel-conditions with |beta| > 50% and the correlation of the run with the mean of the
other runs (voxels in the top 5% of the robust response amplitude). Usage: python3 kay_betaqc.py PREP_DIR"""
import sys, glob, json, numpy as np, nibabel as nib
out = {}
for wd in sorted(glob.glob(f"{sys.argv[1]}/sub-*/ring*")):
    m = np.asarray(nib.load(f"{wd}/mask.nii.gz").dataobj) > 0
    X = np.stack([np.asarray(nib.load(f"{wd}/runbeta_ecc{k}.nii.gz").dataobj, np.float32)[m] for k in range(1, 7)], 1)   # vox cond run
    big = (np.abs(X) > 50).sum((0, 1))
    med = np.median(X, 2); amp = np.abs(med).mean(1); top = amp > np.percentile(amp, 95)
    R = X.shape[2]; r = []
    for i in range(R):
        o = np.delete(X, i, 2).mean(2)
        r.append(float(np.corrcoef(X[top, :, i].ravel(), o[top].ravel())[0, 1]))
    key = wd.split("/")[-2] + "/" + wd.split("/")[-1]; out[key] = dict(n_big=big.tolist(), r_loo=[round(x, 3) for x in r])
    print(key, "|beta|>50%:", big.tolist(), "r(run, others):", [round(x, 2) for x in r], flush=True)
json.dump(out, open(f"{sys.argv[1]}/beta_qc.json", "w"), indent=1)
