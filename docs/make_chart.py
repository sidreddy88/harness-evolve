"""Draws docs/results.png from the held-out numbers in the README (agent-platform's
diagnosis agent, SWE-bench Verified, 66 cases from repositories the optimizer never saw,
DeepSeek-V4.1-Flash). Run: python docs/make_chart.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

LABELS = ["Original\n1 run", "Original\nbest of 2 runs*", "Evolved\n1 run"]
ACC = [78.8, 89.9, 89.9]
COST = [0.033, 0.066, 0.040]
COLORS = ["#9aa5b1", "#9aa5b1", "#2563eb"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

b = a1.bar(LABELS, ACC, color=COLORS, width=0.6)
a1.set_ylim(0, 100)
a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
a1.set_title("Held-out accuracy", fontsize=11, loc="left")
for rect, v in zip(b, ACC):
    a1.text(rect.get_x() + rect.get_width() / 2, v / 2, f"{v:.1f}%", ha="center", fontsize=10,
            color="white", fontweight="bold")
a1.errorbar(2, 89.9, yerr=[[89.9 - (78.8 + 6.1)], [(78.8 + 16.2) - 89.9]], fmt="none",
            ecolor="#1e3a8a", capsize=4, lw=1)

b = a2.bar(LABELS, COST, color=COLORS, width=0.6)
a2.set_ylim(0, 0.08)
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
a2.set_title("Cost per task", fontsize=11, loc="left")
for rect, v in zip(b, COST):
    a2.text(rect.get_x() + rect.get_width() / 2, v / 2, f"${v:.3f}", ha="center", fontsize=10,
            color="white", fontweight="bold")

fig.text(0.01, 0.01, "66 SWE-bench Verified cases from repositories the optimizer never saw; DeepSeek-V4.1-Flash. "
         "Error bar: 95% CI of the gain over one run.\n*Best of 2 assumes a perfect picker between the two answers.",
         fontsize=7.5, color="#52606d")
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = Path(__file__).parent / "results.png"
fig.savefig(out)
print(out)
