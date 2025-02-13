import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--src", type=str, help="Path to the source directory containing the entropy.json files")
args = parser.parse_args()

f = open(os.path.join(args.src, "entropy.json"))
entropy_un = json.load(f)
lab_to_name = json.load(open(os.path.join(args.src, "lab_to_name.json")))

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

fontsize = 20

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

for i in range(len(lab_to_name)):

    plt.figure(figsize=(10, 5))

    genus = lab_to_name[str(i)]
    vals = np.array(entropy_un[genus])
    vals = vals[~np.isnan(vals)]
    x = np.log(vals)
    y = np.log(entropy_lab[genus])
    xmin = np.min((x.min(), y.min()))
    xmax = np.max((x.max(), y.max()))
    xlim = (xmin, xmax)
    e = entropy_lab[lab_to_name[str(i)]]
    q = np.log(np.quantile(e, 1 - args.alpha))
    
    plt.hist(x, bins=100, density=True, alpha=1.0, color=colors[0], label="Unlabelled data")
    plt.hist(y, bins=10, alpha=0.8, density=True, color=colors[i + 1], label=f"{genus}")
    plt.axvline(q, color='r', linestyle='--', linewidth=3, label=f"{args.alpha:.0%} quantile")
    plt.xlabel("log Entropy", fontsize=fontsize)
    plt.xlim(xlim)
    plt.xticks(fontsize=fontsize)
    plt.yticks([])
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    fname = f"{genus}_entropy_distribution.png"
    plt.savefig(os.path.join(args.src, fname), dpi=300)
    plt.close()

