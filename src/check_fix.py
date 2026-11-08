"""
Sanity check for the sampling-rate fix.

If the fix is right, two things should be true afterwards:
  1. every viewing of a clip lasts about as long as the clip, for both groups
  2. the occipital alpha bump (eyes-open, ~8-12 Hz) sits at the same place in both groups
Figure: figures/01_sampling_rate_fix.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 128
meta = pd.read_csv("data/trials.csv")
eeg = np.load("data/eeg128.npz")
keep = meta[meta.keep]

print("length spread within a clip (median abs deviation, seconds):",
      round((keep.seconds - keep.clip_seconds).abs().median(), 2))

def occipital_spectrum(rows):
    specs = []
    for r in rows.itertuples():
        f, p = welch(eeg[f"{r.pid}_{r.emo}"][:, [6, 7]], fs=FS, nperseg=FS * 4, axis=0)
        specs.append(np.log10(p.mean(1)))
    return f, np.array(specs)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for fs, color in [(256, "C0"), (128, "C3")]:
    rows = keep[keep.person_fs == fs]
    f, s = occipital_spectrum(rows.sample(400, random_state=0))
    m = (f >= 2) & (f <= 30)
    # take the 1/f trend out so the alpha bump is easy to see
    trend = np.polyval(np.polyfit(np.log10(f[m]), s.mean(0)[m], 1), np.log10(f[m]))
    resid = s.mean(0)[m] - trend
    ax[0].plot(f[m], resid, color=color, label=f"recorded at {fs} Hz ({rows.pid.nunique()} people)")
    alpha_range = (f[m] >= 6) & (f[m] <= 16)        # look for the bump where alpha can live
    print(fs, "Hz group: alpha peak at", f[m][alpha_range][np.argmax(resid[alpha_range])], "Hz")
    ax[1].hist(rows.length_ratio, bins=40, range=(0.8, 1.2), color=color, alpha=.6, label=f"{fs} Hz group")
ax[0].axvspan(8, 13, color="0.9", zorder=0)
ax[0].set(title="After the fix: occipital spectrum, 1/f removed", xlabel="Hz", ylabel="log power above trend")
ax[0].legend(fontsize=8)
ax[1].set(title="After the fix: recording length / clip length", xlabel="ratio")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/01_sampling_rate_fix.png", dpi=120)
