import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--src", type=str, help="Path to the source directory containing the entropy.json files")
parser.add_argument("--fit_to_normal", type=bool, default=False)
args = parser.parse_args()

f = open(os.path.join(args.src, "entropy.json"))
entropy_un = json.load(f)
f.close()

f = open(os.path.join(args.src, "ref_entropy.json"))
entropy_lab = json.load(f)
f.close()

lab_to_name = json.load(open(os.path.join(args.src, "lab_to_name.json")))

fontsize = 20

#colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
cmap = plt.get_cmap("tab20")

for i in range(len(lab_to_name)):

    plt.figure(figsize=(10, 5))

    genus = lab_to_name[str(i)]
    vals = np.array(entropy_un[genus])
    vals = np.concatenate([entropy_un[k] for k in entropy_un.keys()])
    vals = vals[~np.isnan(vals)]
    x = np.log(vals)
    y = np.log(entropy_lab[genus])
    xmin = np.min((x.min(), y.min()))
    xmax = np.max((x.max(), y.max()))
    xlim = (xmin, xmax)
    e = entropy_lab[lab_to_name[str(i)]]
    q = np.log(np.quantile(e, 1 - args.alpha))
    if args.fit_to_normal == True:
        xax = np.linspace(xmin, xmax, 100)
        yax = norm.pdf(xax, loc=np.mean(x), scale=np.std(x))
        plt.fill_between(xax, yax, alpha=1.0, color=cmap(0), label="Unlabelled data")
        yax = norm.pdf(xax, loc=np.mean(y), scale=np.std(y))
        plt.fill_between(xax, yax, alpha=0.8, color=cmap(2), label=f"{genus}")
        q = norm.ppf(1 - args.alpha, loc=np.mean(y), scale=np.std(y))
    else:
        plt.hist(x, bins=100, density=True, alpha=1.0, label="Unlabelled data", color=cmap(0))
        plt.hist(y, bins=10, alpha=0.8, density=True, label=f"{genus}", color=cmap(2))
    plt.axvline(q, color='r', linestyle='--', linewidth=3, label=f"{args.alpha:.0%} quantile")
    plt.xlabel("log Entropy", fontsize=fontsize)
    plt.xlim(xlim)
    plt.xticks(fontsize=fontsize)
    plt.ylim(bottom=0, top=5)
    plt.yticks([])
    #plt.legend(fontsize=fontsize)
    plt.tight_layout()
    fname = f"{genus}_entropy_distribution.jpg"
    plt.savefig(os.path.join(args.src, fname), dpi=300)
    plt.close()

