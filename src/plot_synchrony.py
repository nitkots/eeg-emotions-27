"""Figures 03 and 04 for the synchrony analysis."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res = pd.read_csv("results/synchrony_by_clip.csv").sort_values("isc")
summ = json.load(open("results/synchrony_summary.json"))
tcs = json.load(open("results/synchrony_timecourses.json"))

# --- 03: synchrony per clip, against both controls -------------------------
fig, ax = plt.subplots(figsize=(7, 7.5))
y = np.arange(len(res))
ax.barh(y, res.isc, color="#2a6", label="same clip (real)")
ax.scatter(res.null_95, y, marker="|", s=120, color="k", label="scrambled timing, 95th pct")
ax.scatter(res.wrong_clip, y, marker="x", s=25, color="#b33", label="different clip (control)")
ax.set_yticks(y, res.emotion, fontsize=8)
ax.axvline(0, color="0.6", lw=.8)
ax.set_xlabel("inter-subject correlation of band power (leave-one-out)")
ax.set_title("How much do brains move together on each clip?")
ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/03_synchrony_by_clip.png", dpi=120)

# --- 04: synchrony vs decodability, and when it happens ---------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].scatter(res.isc, res.recall * 100, color="#2a6")
for r in res.itertuples():
    ax[0].annotate(r.emotion, (r.isc, r.recall * 100), fontsize=7, xytext=(3, 2), textcoords="offset points")
rho, p = summ["isc_vs_recall"]
ax[0].set(xlabel="synchrony across viewers (ISC)", ylabel="honest classifier recall (%)",
          title=f"Synchrony vs decodability: weak trend, not significant (rho={rho:.2f}, p={p:.2f})")
ax[0].axhline(100 / 27, ls="--", color="k", lw=.8)
ax[0].spines[["top", "right"]].set_visible(False)

top = res.sort_values("isc", ascending=False).emotion.head(3).tolist()
for name, c in zip(top, ["C0", "C1", "C2"]):
    t = np.arange(len(tcs[name])) + 3 + 5          # trimmed 3 s; window centre is +5 s
    ax[1].plot(t, tcs[name], color=c, label=name)
    k = int(np.argmax(tcs[name]))
    ax[1].scatter(t[k], tcs[name][k], color=c, zorder=3)
    ax[1].annotate(f"{t[k]} s", (t[k], tcs[name][k]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[1].set(xlabel="seconds into clip (centre of 10 s window)", ylabel="ISC",
          title="When in the clip do people sync? (top 3 clips)")
ax[1].legend(fontsize=8, frameon=False)
ax[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/04_synchrony_vs_decoding_and_time.png", dpi=120)
print("peak times:", {n: int(np.argmax(tcs[n])) + 8 for n in top})
