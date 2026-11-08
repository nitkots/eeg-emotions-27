"""
Load the raw EEGEmotions-27 recordings and fix the sampling-rate problem.

The dataset README says every file is 256 Hz. That isn't true for everyone:
the Emotiv headset notches out power-line noise at 50 and 60 Hz, and for about
half the participants those notches show up at 100 and 120 Hz when you assume
256 Hz. That only happens if the real rate was 128 Hz.

So instead of trusting the README, we look for the notch in each file and
decide the rate from the data itself. Then everyone gets put on a common
128 Hz grid and band-pass filtered.

Output: data/eeg128.npz with one float32 array per trial (samples x 14 channels)
plus a small metadata table (data/trials.csv).
"""
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, sosfiltfilt, welch

CLAIMED_FS = 256          # what the README says
TARGET_FS = 128           # common rate we resample everyone to
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDPASS = butter(4, [1, 40], btype="band", fs=TARGET_FS, output="sos")


def notch_depth(freqs, power, f0):
    """How far power at f0 drops below its neighbours (in log10 units)."""
    at = np.log10(power[np.argmin(abs(freqs - f0))])
    side = np.log10(power[(abs(freqs - f0) > 3) & (abs(freqs - f0) < 6)]).mean()
    return side - at


def detect_true_fs(x):
    """Return 256 if the 50 Hz notch is where it should be, 128 if it sits at 100 Hz."""
    f, p = welch(x - x.mean(0), fs=CLAIMED_FS, nperseg=512, axis=0)
    p = p.mean(1)
    return 256 if notch_depth(f, p, 50) > notch_depth(f, p, 100) else 128


def load_one(path):
    name = os.path.basename(path)
    pid, emo = map(int, re.match(r"(\d+)_(\d+)\.0\.txt", name).groups())
    x = pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy(np.float64)
    if x.ndim != 2 or len(x) < CLAIMED_FS * 4:          # one file in the set is basically empty
        return dict(pid=pid, emo=emo, ok=False)
    fs = detect_true_fs(x)
    if fs == 256:
        x = decimate(x, 2, axis=0, zero_phase=True)      # anti-aliased 256 -> 128
    x = sosfiltfilt(BANDPASS, x - x.mean(0), axis=0).astype(np.float32)
    return dict(pid=pid, emo=emo, ok=True, detected_fs=fs,
                seconds=len(x) / TARGET_FS, data=x)


def main(raw_dir, out_dir):
    files = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".txt"))
    with Pool(2) as pool:
        trials = pool.map(load_one, files, chunksize=8)

    meta = pd.DataFrame([{k: v for k, v in t.items() if k != "data"} for t in trials])
    good = [t for t in trials if t["ok"]]

    # One rate per person: take the majority vote across their 27 files
    meta["person_fs"] = meta.groupby("pid")["detected_fs"].transform(lambda s: s.mode().iloc[0])
    print("files where the detector disagreed with the person's majority:",
          int((meta.ok & (meta.detected_fs != meta.person_fs)).sum()))

    # After the fix, every viewing of the same clip should last about the same time.
    # Anything far off (a recording that kept running, etc.) gets flagged.
    meta["clip_seconds"] = meta.groupby("emo")["seconds"].transform("median")
    meta["length_ratio"] = meta.seconds / meta.clip_seconds
    meta["keep"] = meta.ok & meta.length_ratio.between(0.9, 1.1)
    print(meta.groupby("person_fs").pid.nunique().rename("people per detected rate"))
    print("trials kept:", int(meta.keep.sum()), "of", len(meta))

    os.makedirs(out_dir, exist_ok=True)
    meta.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    np.savez(os.path.join(out_dir, "eeg128.npz"),
             **{f"{t['pid']}_{t['emo']}": t["data"] for t in good})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
