"""
Figure 00: the evidence that some recordings aren't really 256 Hz.

Takes the raw files exactly as given, assumes 256 Hz like the README says,
and plots the occipital spectrum for people whose recordings are full length
vs half length. The headset's power-line notches (50 and 60 Hz) should sit
at 50 and 60. For the half-length people they land at 100 and 120.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

RAW = sys.argv[1]
CLAIMED_FS = 256
CLIP = 10            # calmness; any clip works, this one is 42 s long
meta = pd.read_csv("data/trials.csv")


def spectrum(pid):
    x = pd.read_csv(os.path.join(RAW, f"{pid}_{CLIP}.0.txt"), sep=r"\s+", header=None).to_numpy(float)
    f, p = welch(x - x.mean(0), fs=CLAIMED_FS, nperseg=512, axis=0)
    return f, np.log10(p[:, [6, 7]].mean(1))                 # O1, O2


groups = {
    "full-length recordings": meta[(meta.person_fs == 256)].pid.unique()[:12],
    "half-length recordings": meta[(meta.person_fs == 128)].pid.unique()[:12],
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, (title, pids) in zip(ax, groups.items()):
    specs = []
    for pid in pids:
        f, s = spectrum(pid)
        specs.append(s)
        a.plot(f, s, color="0.75", lw=.6)
    a.plot(f, np.mean(specs, 0), color="C3", lw=2, label="mean of 12 people")
    for hz in (50, 60, 100, 120):
        a.axvline(hz, color="C0", ls=":", lw=.8)
    a.set(title=title, xlabel="frequency (Hz), assuming 256 Hz like the README")
    a.legend(fontsize=8)
ax[0].set_ylabel("log power at O1/O2")
fig.suptitle("Power-line notches at 50/60 Hz (left) show up at 100/120 Hz (right): those people were really recorded at 128 Hz",
             fontsize=10)
plt.tight_layout()
plt.savefig("figures/00_sampling_rate_evidence.png", dpi=120)
