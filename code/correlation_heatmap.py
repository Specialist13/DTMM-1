from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent.parent
chartsDirectory = root / "charts"
chartsDirectory.mkdir(exist_ok=True)
data = pd.read_csv(root / "A29.csv", encoding="utf-8-sig").drop(columns="class")
# Parse raw numeric values; missing/text values are ignored pairwise.
data = data.replace(r"\s*a\.u\.\s*$", "", regex=True).apply(pd.to_numeric, errors="coerce")
correlation = data.corr(method="pearson").abs()

figure, axis = plt.subplots(figsize=(14, 12))
colormap = plt.get_cmap("Reds").copy()
colormap.set_bad("white")
upperTriangle = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
matrix = np.ma.array(correlation.to_numpy(), mask=upperTriangle)
image = axis.imshow(matrix, cmap=colormap, vmin=0, vmax=1, aspect="auto")
positions = range(len(correlation.columns))
axis.set_xticks(positions, labels=correlation.columns, rotation=90)
axis.set_yticks(positions, labels=correlation.index)
axis.set(xlabel="Požymis", ylabel="Požymis")
figure.colorbar(image, ax=axis, label="Pirsono koreliacijos modulis", shrink=0.8)
figure.tight_layout()
figure.savefig(chartsDirectory / "A29_correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(figure)
print("Saved: charts/A29_correlation_heatmap.png")
