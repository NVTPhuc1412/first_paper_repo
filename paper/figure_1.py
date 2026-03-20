import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Data from Table 2 ---
models = [
    "None + Anomaly Transformer",
    "PatchTST + Anomaly Transformer",
    "TimesNet + TranAD",
    "PatchTST + TranAD",
    "iTransformer + TranAD",
    "None + TranAD",
    "TimesNet + Anomaly Transformer",
    "iTransformer + Anomaly Transformer",
]

conditions = [
    "Synthetic\nEasy",
    "Synthetic\nMedium",
    "Synthetic\nHard",
    "Real\nEasy",
    "Real\nMedium",
    "Real\nHard",
]

data = np.array(
    [
        [0.6978, 0.6570, 0.5103, 0.6738, 0.6280, 0.5184],
        [0.6427, 0.6204, 0.4939, 0.6505, 0.5987, 0.5171],
        [0.8035, 0.6293, 0.2810, 0.6478, 0.5850, 0.4973],
        [0.7391, 0.5144, 0.1923, 0.6819, 0.5900, 0.4846],
        [0.7718, 0.5203, 0.1178, 0.6510, 0.5620, 0.4504],
        [0.7308, 0.4804, 0.0888, 0.6829, 0.5858, 0.4898],
        [0.6348, 0.5501, 0.3857, 0.5736, 0.4876, 0.3435],
        [0.5510, 0.4892, 0.3558, 0.5859, 0.5110, 0.3571],
    ]
)

df = pd.DataFrame(data, index=models, columns=conditions)

# --- Plot ---
fig, ax = plt.subplots(figsize=(11, 6.5))

sns.heatmap(
    df,
    ax=ax,
    annot=True,
    fmt=".4f",
    annot_kws={"size": 12, "weight": "bold"},
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "F1 Score", "shrink": 0.8},
)

# Column headers on top
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", labelsize=10, length=0)
ax.tick_params(axis="y", labelsize=9.5, length=0)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
out = "./figure1_f1_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out}")
