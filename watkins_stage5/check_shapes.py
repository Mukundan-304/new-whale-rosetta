import numpy as np, pandas as pd
from pathlib import Path

root = Path(".")
for tag in ["watkins_stage4", "watkins_stage10"]:
    for split in ["train", "validation", "test"]:
        d = root / tag / split
        e, m = d / "embeddings.npy", d / "metadata.csv"
        es = np.load(e, mmap_mode="r").shape if e.exists() else None
        ms = pd.read_csv(m).shape if m.exists() else None
        print(tag, split, "emb", es, "meta", ms)

for split in ["train", "validation", "test"]:
    m = pd.read_csv(root / "watkins_stage4" / split / "metadata.csv")
    runs = int((m["feature_path"] != m["feature_path"].shift()).sum())
    print(split, "rows", len(m), "feature_path runs", runs, "unique", m["feature_path"].nunique())
    print(split, "species in first 500 rows:", m.head(500)["species_code"].nunique(),
          "of", m["species_code"].nunique())
print(m.columns.tolist())