import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = glob.glob('./postprocessing/results/test run all slides pytorch classifier alpha 0.5/stats' + "/*.csv")
files.sort()

genus_counts = {}

for file in files:
    df = pd.read_csv(file)
    count = np.unique(df["label"].values, return_counts=True)
    genus_counts[os.path.basename(file)] = dict(zip(count[0], count[1]))

df = pd.DataFrame(genus_counts).T.fillna(0).astype(int)
df["filename"] = df.index # add index
df.index = range(len(df)) # reset index

df["total_count"] = pd.read_csv('./preprocessing/hdf5/counts.csv')["count"].values # add total count

d = df["filename"].values
d = [x.split(" ")[1] for x in d]
d = [x.split(".")[0] for x in d]
df["depth"] = d

df["alpha"] = pd.read_csv('./postprocessing/results/test run all slides pytorch classifier alpha 0.5/alpha.csv').iloc[:, 1].values

# smooth the data
for i in [0, 4, 11, 14]:
    y = (df[i] / (1 - df["alpha"])).rolling(window=1).mean()

    plt.figure(figsize=(20, 10))
    plt.plot(df["depth"], y, label="Genus count", marker="o")
    plt.xticks(df["depth"][::4], rotation=45)
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.title("Genus distribution")
    plt.savefig(f"./postprocessing/results/test run all slides pytorch classifier alpha 0.5/Class{i}_distribution.png")
    plt.close()