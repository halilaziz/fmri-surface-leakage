# Environment

## Containers (Apptainer)

| image | used for | pull |
|---|---|---|
| `pennlinc/xcp_d:26.0.2` | all Python analysis (Python 3.12, numpy, scipy, nibabel, matplotlib; nilearn for the fsaverage5 surfaces of `theory_sims/run_banks_orient.py`) and Connectome Workbench `wb_command` 2.0.1 | `apptainer pull xcp_d.sif docker://pennlinc/xcp_d:26.0.2` |
| `nipreps/fmriprep:24.1.1` | FSL tools (MCFLIRT, FLIRT with BBR, fugue, fsl_prepare_fieldmap, bet2), FreeSurfer 7 (bbregister) and Python | `apptainer pull fmriprep.sif docker://nipreps/fmriprep:24.1.1` |

Python was run inside the containers with `apptainer exec`, binding the data roots; on a server with NFS storage, `--no-home`
and a `/tmp` bound to the workspace keep everything inside it. FreeSurfer tools need a FreeSurfer licence (`FS_LICENSE`).

## Known pitfalls

- **Concurrent Workbench mapping.** Concurrent `wb_command -volume-to-surface-mapping` calls on one node crash with SIGBUS. The
  code serialises them with a lock file (`GEO_WB_LOCK`). On file systems without lock support (some NFS mounts), set
  `GEO_WB_LOCK=none` and run one mapping job per node.
- **Threaded GLMs.** Do not run the per-run GLMs (`numpy.linalg.lstsq`) in parallel Python threads. In one container this silently
  corrupted whole runs. The released code runs them sequentially; `kay_betaqc.py` checks every run.
- **Field-map sign.** fugue's unwarp direction is chosen per session by the registration cost of both signs (`kay_undistort.py`).

## Compute used

- **An HPC cluster** (TRUBA, TÜBİTAK ULAKBİM; SLURM, 8–28 cores per job, ≤ 4 h).
- **A lab server** (112 cores, 1 TB RAM; Apptainer).

`run/run_statistics.sh` needs python3 with numpy, scipy and matplotlib; it regenerates the tests, tables and result figures from
the result files in a few minutes.

Typical costs:
- HCP operator and leakage: about 30 min per subject;
- HCP week-3 fits: about 1–2 h per subject for the full prior grid;
- StudyForrest and ds002702 within-session fits: 20–60 min per subject, session and factor;
- contamination maps: 4–20 min per cell.
- NYU 3T (week 17): preprocessing about 19 min per subject; 4.5–6 h per tuning subject for the full prior grid and 28–77 min per
  test subject at the chosen settings (12 subjects in parallel on the lab server, 6 threads each); group maps under a minute.

Peak memory is about 15 GB per job.
