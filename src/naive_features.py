"""
The "naive" version: spectra computed the way the README says to, at 256 Hz
for everyone. benchmark.py uses this as the do-what-most-people-would-do baseline.

Output: data/naive_psd256.npz (per trial: channels x frequencies, 0-45 Hz)
"""
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

FS = 256                                   # trusting the README
BANDPASS = butter(4, [1, 45], btype="band", fs=FS, output="sos")


def one(path):
    pid, emo = map(int, re.match(r"(\d+)_(\d+)\.0\.txt", os.path.basename(path)).groups())
    x = pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy(np.float32)
    if x.ndim != 2 or len(x) < FS * 4:
        return None
    y = sosfiltfilt(BANDPASS, x - x.mean(0), axis=0)
    f, p = welch(y, fs=FS, nperseg=FS * 2, axis=0)
    return pid, emo, p[f <= 45].T.astype(np.float32)


if __name__ == "__main__":
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    files = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".txt"))
    with Pool(2) as pool:
        res = [r for r in pool.map(one, files, chunksize=8) if r is not None]
    np.savez_compressed(os.path.join(out_dir, "naive_psd256.npz"),
                        pid=np.array([r[0] for r in res]), emo=np.array([r[1] for r in res]),
                        psd=np.stack([r[2] for r in res]), freqs=np.arange(0, 45.5, 0.5))
