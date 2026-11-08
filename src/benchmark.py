"""
How much of "decoding 27 emotions" is actually decoding brain activity?

Every emotion in this dataset comes from one specific video clip, so
"emotion" and "which clip, how long was it" are the same thing. Here I train
the same kind of classifier with different inputs and see what's left once
the shortcuts are gone. All tests hold out whole people (5-fold by participant),
so the model is always scored on brains it has never seen.

  A  recording length only                          -> pure shortcut
  B  naive: trust the README (256 Hz), whole trial  -> what most people would do
  C  naive + length                                 -> what a feature dump often leaks
  D  corrected rate, first 18 s only, no length     -> the honest version

Extra test: train on the true-256 Hz people, test on the true-128 Hz people,
with and without the rate fix. If the bug matters, the naive version
should fall apart across groups.
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 128
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
SECONDS = 18                     # shortest clip is ~19 s, so every trial has 18 s
rng_seed = 0

meta = pd.read_csv("data/trials.csv")
meta = meta[meta.keep].reset_index(drop=True)
eeg = np.load("data/eeg128.npz")


def band_powers(f, p):
    """log power per band per channel -> flat vector (5 bands x 14 channels)."""
    return np.concatenate([np.log10(p[(f >= lo) & (f < hi)].mean(0)) for lo, hi in BANDS.values()])


# --- naive features: what you get if you believe the README -----------------
# Reuse the raw files' spectra computed earlier at the claimed 256 Hz.
naive = np.load("data/naive_psd256.npz")
naive_index = {(p, e): i for i, (p, e) in enumerate(zip(naive["pid"], naive["emo"]))}
fn = naive["freqs"]

def naive_features(row):
    p = naive["psd"][naive_index[(row.pid, row.emo)]].T      # freqs x channels
    return band_powers(fn, p)

# --- corrected features: fixed rate, fixed length, per-person normalised -----
def honest_features(row):
    x = eeg[f"{row.pid}_{row.emo}"][: SECONDS * FS]
    f, p = welch(x, fs=FS, nperseg=FS * 2, axis=0)
    return band_powers(f, p)

y = meta.emo.to_numpy()
groups = meta.pid.to_numpy()
length = meta.seconds.to_numpy()[:, None]
Xn = np.array([naive_features(r) for r in meta.itertuples()])
Xh = np.array([honest_features(r) for r in meta.itertuples()])

def per_person_z(X):
    """Remove each person's overall level so the model can't key on 'whose head'."""
    out = X.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    return out

Xn_z, Xh_z = per_person_z(Xn), per_person_z(Xh)

models = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)),
    "forest": lambda: RandomForestClassifier(400, min_samples_leaf=3, n_jobs=2, random_state=rng_seed),
}

def cv_accuracy(X, model):
    accs = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = models[model]().fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))

setups = {
    "A  length only": length,
    "B  naive (README rate, whole trial)": Xn_z,
    "C  naive + length": np.hstack([Xn_z, length]),
    "D  corrected rate, first 18 s, no length": Xh_z,
}
results = {}
for name, X in setups.items():
    for model in models:
        m, s = cv_accuracy(X, model)
        results[f"{name} | {model}"] = (round(m, 3), round(s, 3))
        print(f"{name:45s} {model:9s} {m:.3f} ± {s:.3f}")
print("chance:", round(1 / 27, 3))

# --- does the rate bug break transfer between the two groups? --------------
# Use raw (not per-person z-scored) features here: the z-scoring would hide
# exactly the frequency shift we want to expose.
src = meta.person_fs.to_numpy() == 256
tgt = ~src
transfer = {}
for name, X in [("naive", Xn), ("corrected", Xh)]:
    clf = models["logistic"]().fit(X[src], y[src])
    acc = (clf.predict(X[tgt]) == y[tgt]).mean()
    # also: how different do the two groups look in alpha vs beta?
    alpha = X[:, 2 * 14:3 * 14].mean(1); beta = X[:, 3 * 14:4 * 14].mean(1)
    gap = float(np.median((alpha - beta)[src]) - np.median((alpha - beta)[tgt]))
    transfer[name] = dict(train256_test128_acc=round(float(acc), 3), alpha_minus_beta_group_gap=round(gap, 3))
    print(f"transfer {name:9s}: acc {acc:.3f}, alpha-beta gap between groups {gap:+.3f} (log10)")

json.dump(dict(cv=results, transfer=transfer, chance=1 / 27), open("results/benchmark.json", "w"), indent=2)
