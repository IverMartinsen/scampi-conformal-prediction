import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from postprocessing.utils import lab_to_name

path_to_counts = './data/NO 6407-6-5/hdf5/counts.csv'

stats = pd.read_csv('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/results/NO 15-9-19 A_alpha_0.05/stats.csv')
#lab_to_name = json.load(open('data/BaileyLabels/imagefolder/lab_to_name.json'))
source = np.unique(stats["source"].values)

genus_counts = {}

for src in source:
    df = stats[stats["source"] == src]
    count = np.unique(df["genus"].values, return_counts=True)
    genus_counts[src] = dict(zip(count[0], count[1]))

df = pd.DataFrame(genus_counts).T.fillna(0).astype(int)

classes = df.columns

#a = pd.read_csv(path_to_counts)
#a.index = a["filename"].values # set index
#a = a.drop(columns=["filename"])

# merge df with a
#df = df.merge(a, left_index=True, right_index=True, how="outer") # outer join
#df = df.fillna(0).astype(int)

d = df.index.values
d = [x.split(" ")[2] for x in d]
d = [x.split(".")[0] for x in d]
df["depth"] = [float(x) for x in d]

for i in [0, 4, 11, 14]:
    # check if col exists
    if i not in df.columns:
        df[i] = 0

# reorder columns
cols = [0, 4, 11, 14, "count", "depth"]
cols = [1, 2, 3, 4]
df = df[cols]

# smooth the data
fontsize = 18
fdr = {0: 0.18, 4: 0.26, 11: 0.02, 14: 0.01}
alpha = 0.95

# default matplot colormap

span = (3780, 3920)
span = (3190, 4140)

df = df.loc[(df["depth"] >= span[0]) & (df["depth"] <= span[1])]

cmap = plt.get_cmap("tab20")
iterable = iter(cmap.colors)
#for i in [0, 4, 11, 14]:
for c in classes:

    #y = (df[i] * (1 - fdr[i]) / (1 - alpha)).rolling(window=1).mean()
    y = (df[c]).rolling(window=1).mean()
    
    plt.figure(figsize=(20, 10))
    #plt.plot(df["depth"], y, label="Genus count", marker="o")
    plt.bar(df["depth"], y, label="Genus count", color=next(iterable), width=3)
    #plt.xticks(df["depth"][::4], rotation=45, fontsize=fontsize)
    plt.xticks(np.arange(span[0], span[1], 20), rotation=45, fontsize=fontsize)
    plt.xlabel("Depth", fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)
    plt.title(f"{c} distribution", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'{c}_distribution.png')
    plt.close()


# joint plot
tmp = df.copy()
tmp.index = tmp.depth
tmp.drop(columns=["depth", "count"], inplace=True)
for i in tmp.columns:
    tmp[i] = (tmp[i] * (1 - fdr[i]) / (1 - alpha))
tmp.columns = [lab_to_name[i] for i in tmp.columns]

tmp.plot(kind="bar", stacked=True, figsize=(20, 10), fontsize=4)
plt.savefig(f'/Users/ima029/Desktop/NO 6407-6-5/postprocessing/results/NO 6407-6-5_alpha_{alpha}/joint_distribution.png')
plt.close()