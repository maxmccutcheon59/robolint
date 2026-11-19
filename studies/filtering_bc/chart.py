"""Bar chart of policy success by training-data version. Run after study.py."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
runs = json.loads((HERE / "results.json").read_text())
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "filtering-study.png"

CONDS = [
    ("raw", "Raw data", "#2a78d6"),
    ("robolint-err", "robolint: drop errors", "#eb6834"),
    ("robolint-all", "robolint: drop errors + warnings", "#1baf7a"),
    ("oracle", "Only clean, successful demos", "#eda100"),
]
MIXES = [("light", "Light faults\n(13% of episodes)"), ("heavy", "Heavy faults\n(36% of episodes)")]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
width, gap = 0.19, 0.012
for m, (mix, _) in enumerate(MIXES):
    rs = [r for r in runs if r["mix"] == mix]
    for c, (key, label, color) in enumerate(CONDS):
        v = np.array([x for r in rs for x in r["scores"][key]]) * 100
        x = m + (c - 1.5) * width
        ax.bar(x, v.mean(), width - 2 * gap, color=color, label=label if m == 0 else None, zorder=2)
        ax.errorbar(x, v.mean(), yerr=v.std(), color=MUTED, lw=1, capsize=2.5, zorder=3)
        ax.text(x, v.mean() + v.std() + 1.2, f"{v.mean():.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(range(len(MIXES)), [lbl for _, lbl in MIXES], color=INK)
ax.set_ylabel("Task success rate (300 test setups)", color=MUTED)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, length=0)
ax.set_title("Policy success by training data", loc="left", color=INK, fontsize=12, pad=26)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE)
print(out)
