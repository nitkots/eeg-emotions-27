"""
Is the synchrony just eyes?

Frontal channels (AF3, AF4, F7, F8) sit right above the eyes, so blinks and eye
movements leak into them. People tend to blink at the same moments in a video
(scene cuts), so frontal synchrony could be mostly eyes, not brain.

Posterior channels (P7, O1, O2, P8) are over visual cortex and far from the eyes.
If those still sync above the controls, there's a shared brain response, not just
shared blinking. Same method as synchrony.py, just restricted to each region.
"""
import json

import numpy as np
import pandas as pd

import synchrony as S          # reuses envelopes, clean, loo_isc, circular_shift, clips

REGIONS = {"frontal (near eyes)": [0, 13, 1, 12], "posterior (visual cortex)": [5, 6, 7, 8]}
N_NULL = 200
rows = []
for emo in range(1, 28):
    E = S.clean(S.clips[emo][1])
    for region, ch in REGIONS.items():
        Er = E[:, :, ch]
        obs = S.loo_isc(Er).mean()
        null = np.array([S.loo_isc(S.circular_shift(Er)).mean() for _ in range(N_NULL)])
        rows.append(dict(emotion=S.EMOTIONS[emo - 1], region=region, isc=obs,
                         null_95=np.percentile(null, 95), p=(1 + (null >= obs).sum()) / (N_NULL + 1)))
df = pd.DataFrame(rows)


def bh(p):
    """Benjamini-Hochberg: which p-values survive testing 27 clips at once."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


summary = {}
for region, g in df.groupby("region"):
    q = bh(g.p.to_numpy())
    summary[region] = dict(mean_isc=round(g.isc.mean(), 4), mean_null95=round(g.null_95.mean(), 4),
                           clips_significant_fdr05=int((q < 0.05).sum()))
    print(region, summary[region])
df.to_csv("results/synchrony_by_region.csv", index=False)
json.dump(summary, open("results/synchrony_by_region.json", "w"), indent=2)
