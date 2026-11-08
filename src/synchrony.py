"""
Do people's brains move together when they watch the same clip?

This dataset has a property most emotion datasets avoid: all 88 people watched
the exact same clip for each emotion. That's a problem for classification
(see benchmark.py), but it's an opportunity for a different question.

Inter-subject correlation (ISC): for each clip, take each person's
moment-to-moment band power and correlate it with the average of everyone
else. If the clip drives brains in a shared way, the correlation is above
what you get when you scramble the timing.

Steps
  1. band power envelope (Hilbert) for theta, alpha, beta, per channel
  2. average into 1-second bins
  3. leave-one-out ISC per person, averaged over channels -> one number per person per clip
  4. clean-up: drop 3 s at each end and remove linear drift. My first version
     skipped this and got ISC ~0.12, but about half of that was filter edge
     effects and slow drift that line up across people for boring reasons.
  5. two controls:
       - circular shift: scramble each person's timing (200 times)
       - wrong clip: correlate a person on this clip with everyone else on a
         *different* clip. Same people, same headset, same processing, but no
         shared content. If ISC is about the video, this should be ~0.
  6. sliding 10 s window to find *when* in each clip people sync up
Also: do clips that sync brains more turn out to be easier to classify?
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, hilbert, sosfiltfilt, welch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 128
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
EMOTIONS = ["admiration", "adoration", "aesthetic", "amusement", "anger", "anxiety", "awe",
            "awkwardness", "boredom", "calmness", "confusion", "craving", "disgust", "empathic pain",
            "entrancement", "excitement", "fear", "horror", "interest", "joy", "nostalgia", "relief",
            "romance", "sadness", "satisfaction", "sexual desire", "surprise"]
N_NULL = 200
TRIM = 3
rng = np.random.default_rng(0)

meta = pd.read_csv("data/trials.csv")
meta = meta[meta.keep].reset_index(drop=True)
eeg = np.load("data/eeg128.npz")
filters = {b: butter(4, r, btype="band", fs=FS, output="sos") for b, r in BANDS.items()}


def envelopes(x):
    """bands x channels x seconds of log band power, z-scored over time."""
    n_sec = len(x) // FS
    out = []
    for sos in filters.values():
        env = np.abs(hilbert(sosfiltfilt(sos, x, axis=0), axis=0))[: n_sec * FS]
        env = np.log(env.reshape(n_sec, FS, -1).mean(1) + 1e-9)      # 1 s bins
        out.append(((env - env.mean(0)) / (env.std(0) + 1e-9)).T)
    return np.stack(out)


def clean(E):
    """Cut the filter edges off and remove slow linear drift."""
    return detrend(E[..., TRIM:-TRIM], axis=-1)


def loo_isc(E, others_from=None):
    """E: people x bands x channels x time. Each person vs mean of the rest.
    If others_from is given (same people, a different clip), compare against that instead."""
    O = E if others_from is None else others_from
    total = O.sum(0)
    n = len(E)
    r = np.empty(E.shape[:3])
    for i in range(n):
        others = (total - O[i]) / (n - 1)
        a = E[i] - E[i].mean(-1, keepdims=True)
        b = others - others.mean(-1, keepdims=True)
        r[i] = (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
    return r                                                          # people x bands x channels


def circular_shift(E):
    T = E.shape[-1]
    return np.stack([np.roll(e, rng.integers(5, T - 5), axis=-1) for e in E])


# envelopes for every clip, once
clips = {}
for emo in range(1, 28):
    rows = meta[meta.emo == emo]
    T = int(rows.seconds.min())
    clips[emo] = (rows.pid.to_numpy(),
                  np.stack([envelopes(eeg[f"{r.pid}_{r.emo}"])[..., :T] for r in rows.itertuples()]))


def wrong_clip_isc(emo):
    """This clip's people vs. the same people on every other clip, length-matched."""
    pids, E = clips[emo]
    vals = []
    for other in range(1, 28):
        if other == emo:
            continue
        pids2, E2 = clips[other]
        common = np.intersect1d(pids, pids2)
        a = E[np.isin(pids, common)]
        b = E2[np.isin(pids2, common)]
        a = a[np.argsort(pids[np.isin(pids, common)])]
        b = b[np.argsort(pids2[np.isin(pids2, common)])]
        T = min(a.shape[-1], b.shape[-1])
        vals.append(loo_isc(clean(a[..., :T]), others_from=clean(b[..., :T])).mean())
    return float(np.mean(vals))


def main():
    results, timecourses = [], {}
    for emo in range(1, 28):
        rows = meta[meta.emo == emo]
        T = int(rows.seconds.min())
        E = clean(clips[emo][1])
        isc = loo_isc(E)                                     # people x bands x channels
        observed = isc.mean()                                # one number per clip
        null = np.array([loo_isc(circular_shift(E)).mean() for _ in range(N_NULL)])
        p = (1 + (null >= observed).sum()) / (N_NULL + 1)
        wrong = wrong_clip_isc(emo)

        # when does it happen? 10 s sliding window on the across-band, across-channel mean
        win = 10
        tc = []
        for t0 in range(0, E.shape[-1] - win + 1):
            tc.append(loo_isc(E[..., t0:t0 + win]).mean())
        timecourses[EMOTIONS[emo - 1]] = tc

        results.append(dict(
            emotion=EMOTIONS[emo - 1], emo=emo, people=len(rows), seconds=T,
            isc=observed, null_mean=null.mean(), null_95=np.percentile(null, 95), p=p, wrong_clip=wrong,
            **{f"isc_{b}": isc[:, i].mean() for i, b in enumerate(BANDS)},
            isc_frontal=isc[:, :, [0, 13, 1, 12]].mean(),      # AF3 AF4 F7 F8: eye movements live here
            isc_posterior=isc[:, :, [5, 6, 7, 8]].mean(),      # P7 O1 O2 P8: visual cortex
        ))
        print(f"{EMOTIONS[emo-1]:15s} ISC {observed:.4f}  shift-null95 {np.percentile(null,95):.4f}  wrong-clip {wrong:.4f}  p={p:.3f}")

    res = pd.DataFrame(results)

    # --- link to decoding: per-clip recall from the honest classifier -----------
    SECONDS = 18
    def honest_features(r):
        x = eeg[f"{r.pid}_{r.emo}"][: SECONDS * FS]
        f, p = welch(x, fs=FS, nperseg=FS * 2, axis=0)
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
        return np.concatenate([np.log10(p[(f >= lo) & (f < hi)].mean(0)) for lo, hi in bands])
    X = np.array([honest_features(r) for r in meta.itertuples()])
    for g in meta.pid.unique():
        m = (meta.pid == g).to_numpy()
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)),
                             X, meta.emo, groups=meta.pid, cv=GroupKFold(5))
    recall = pd.Series(pred == meta.emo).groupby(meta.emo).mean()
    res["recall"] = res.emo.map(recall)
    rho, p_rho = spearmanr(res.isc, res.recall)
    print(f"\nISC vs decoding recall across clips: Spearman rho={rho:.2f}, p={p_rho:.3f}")
    rho_len, p_len = spearmanr(res.seconds, res.isc)
    print("frontal (eye-movement) vs posterior (visual) ISC, mean over clips:",
          round(res.isc_frontal.mean(), 4), round(res.isc_posterior.mean(), 4))
    print(f"clip length vs ISC: rho={rho_len:.2f}, p={p_len:.3f}")

    res.to_csv("results/synchrony_by_clip.csv", index=False)
    json.dump(dict(isc_vs_recall=[rho, p_rho], length_vs_isc=[rho_len, p_len],
                   n_significant_p05=int((res.p < 0.05).sum())),
              open("results/synchrony_summary.json", "w"), indent=2)
    json.dump(timecourses, open("results/synchrony_timecourses.json", "w"))


if __name__ == "__main__":
    main()
