import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data from Table 3 ---
difficulties = ["Easy", "Medium", "Hard"]
x = np.arange(len(difficulties))

data = {
    "AT Point": [0.374, 0.341, 0.304],
    "TranAD Point": [0.923, 0.747, 0.464],
    "AT Contextual": [0.836, 0.753, 0.468],
    "TranAD Contextual": [0.893, 0.664, 0.265],
}

# --- Style config ---
styles = {
    "AT Point": dict(color="#2563EB", marker="o", linestyle="-", linewidth=2.2),
    "TranAD Point": dict(color="#2563EB", marker="s", linestyle="--", linewidth=2.2),
    "AT Contextual": dict(color="#DC2626", marker="o", linestyle="-", linewidth=2.2),
    "TranAD Contextual": dict(
        color="#DC2626", marker="s", linestyle="--", linewidth=2.2
    ),
}

fig, ax = plt.subplots(figsize=(7, 4.5))

for label, values in data.items():
    s = styles[label]
    ax.plot(
        x,
        values,
        label=label,
        marker=s["marker"],
        color=s["color"],
        linestyle=s["linestyle"],
        linewidth=s["linewidth"],
        markersize=7,
        zorder=3,
    )


# --- Highlight the crossing between Easy and Medium for contextual lines ---
ax.axvspan(0, 1, alpha=0.06, color="#F59E0B", zorder=0)
ax.text(
    0.5,
    0.27,
    "crossing\n(complementarity)",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#B45309",
    style="italic",
    transform=ax.get_xaxis_transform(),
)

# --- Axes formatting ---
ax.set_xticks(x)
ax.set_xticklabels(difficulties, fontsize=11)
ax.set_xlim(-0.25, 2.25)
ax.set_ylim(0.15, 1.02)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel("Difficulty", fontsize=12, labelpad=6)
ax.set_ylabel("Recall", fontsize=12, labelpad=6)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# --- Legend ---
ax.legend(
    loc="upper right",
    fontsize=9,
    framealpha=0.9,
    edgecolor="#CBD5E1",
    title="Method",
    title_fontsize=9,
)

plt.tight_layout()
plt.savefig("./recall_by_difficulty.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → figure2_recall_by_difficulty.png")
