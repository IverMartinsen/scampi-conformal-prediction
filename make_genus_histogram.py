import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = glob.glob("test run all slides/stats/*.csv")
files.sort()

genus_counts = {}

for file in files:
    df = pd.read_csv(file)
    count = np.unique(df["label"].values, return_counts=True)
    genus_counts[os.path.basename(file)] = dict(zip(count[0], count[1]))

df = pd.DataFrame(genus_counts).T.fillna(0).astype(int)
# add index
df["filename"] = df.index
#df = pd.DataFrame(genus_counts.items(), columns=["filename", "count"])

total_counts = pd.read_csv("hdf5/counts.csv")["count"].values
df["total_count"] = total_counts

d = df.index.values
d = [x.split(" ")[1] for x in d]
d = [x.split(".")[0] for x in d]

df["depth"] = d

df["total_count"].mean()

# smooth the data
for i in [0, 4, 11, 14]:
    y = df[i].rolling(window=5).mean()

    #y = df["count"].rolling(window=1).mean()
    #y = (df["count"] * df["total_count"].mean() / df["total_count"]).rolling(window=5).mean()

    plt.figure(figsize=(20, 10))
    plt.plot(df["depth"], y, label="Genus count", marker="o")
    plt.xticks(df["depth"][::4], rotation=45)
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.title("Genus distribution")
    plt.savefig(f"Class{i}_distribution.png")
    plt.close()