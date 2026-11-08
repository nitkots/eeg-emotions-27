"""Figure 02: how much of the accuracy is the shortcut?"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results/benchmark.json"))
labels = ["Recording\nlength only", "Brain, naive\n(README rate)", "Naive brain\n+ length", "Brain, corrected\n(no length)"]
keys = ["A  length only", "B  naive (README rate, whole trial)", "C  naive + length", "D  corrected rate, first 18 s, no length"]
forest = [r["cv"][f"{k} | forest"][0] * 100 for k in keys]
logit = [r["cv"][f"{k} | logistic"][0] * 100 for k in keys]
x = range(len(keys))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([i - .2 for i in x], forest, .4, label="random forest", color=["#b33", "#999", "#b33", "#2a6"])
ax.bar([i + .2 for i in x], logit, .4, label="logistic regression", color=["#d88", "#ccc", "#d88", "#7c9"])
ax.axhline(100 / 27, ls="--", color="k", lw=1)
ax.text(1.5, 100 / 27 + 1, "chance (3.7%)", ha="center", fontsize=8)
for i, v in enumerate(forest):
    ax.text(i - .2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(list(x), labels, fontsize=9)
ax.set_ylabel("27-way accuracy on unseen people (%)")
ax.set_title("Red bars use the recording length. Green is the honest number.")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/02_shortcut_vs_honest.png", dpi=120)
