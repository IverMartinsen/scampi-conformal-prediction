import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf5

















entropy = json.load('/Users/ima029/Desktop/NO 6407-6-5/4-conformal-prediction/results/NO 15-9-19 A_alpha_0.5_20250211114957/entropy.json')


import json
import os
lab_to_name = json.load(open('/Users/ima029/Desktop/NO 6407-6-5/data/BaileyLabels/imagefolder-bisaccate/lab_to_name.json'))

src_models = [
    './training/trained_models/20250210152137_seed1',
    './training/trained_models/20250210155635_seed2',
    './training/trained_models/20250210163107_seed3',
    './training/trained_models/20250210170115_seed4',
    './training/trained_models/20250210173135_seed5',
    './training/trained_models/20250210180149_seed6',
    './training/trained_models/20250210183208_seed7',
    './training/trained_models/20250210190244_seed8',
    './training/trained_models/20250210193252_seed9',
    './training/trained_models/20250210200258_seed10',
]

entropies = []

for path in src_models:
    with open(os.path.join(path, "entropy.json")) as f:
        entropies.append(json.load(f))

entropy_lab = {}
for k in lab_to_name.values():
    entropy_lab[k] = []
    for e in entropies:
        entropy_lab[k] += e[k]








colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
xlim = (np.log(entropy).min(), np.log(entropy).max())

int_to_class = {
    0: "alisocysta",
    4: "bisaccate",
    11: "inaperturopollenites",
    14: "palaeoperidinium",
}

fontsize = 20

q = np.quantile(entropy, 0.05)

t = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
    t[i] = (e > q).mean()




alpha = t.max()

q_class_wise = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
    n = len(e)
    q_class_wise[i] = np.quantile(e, 1 - alpha)


fig, axes = plt.subplots(5, 1, figsize=(20, 20), sharex=True)

ax = axes.flatten()[0]
ax.hist(np.log(entropy), bins=100, density=True, alpha=0.5, color=colors[0], label="Unlabelled data from slide")
ax.axvline(np.log(q), color='r', linestyle='--', label="5% quantile", linewidth=2)
ax.set_xlim(xlim)
ax.set_yticks([])
ax.legend(fontsize=fontsize)
for j, i in enumerate([0, 4, 11, 14]):
    ax = axes.flatten()[j + 1]
    ax.hist(np.log(entropy_lab)[y_lab == i], bins=10, alpha=0.5, density=True, color=colors[j + 1], label=f"{int_to_class[i]}")
    ax.axvline(np.log(q_class_wise[i]), color='r', linestyle='--', linewidth=2, label="adjusted quantile")
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.legend(fontsize=fontsize)
plt.xlabel("log Entropy", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"entropy_distribution{i}.png", dpi=300)
plt.close()
