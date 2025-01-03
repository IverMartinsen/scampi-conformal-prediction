import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

with open('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/trained_models/merged_entropies.json', 'r') as f:
    ent_lab = json.load(f)

with open('...', 'r') as f:
    ent_unl = json.load(f)

classes_of_interest = ['alisocysta', 'bisaccate', 'inaperturopollenites', 'palaeoperidinium']


for class_name in classes_of_interest:

    x = ent_lab[class_name]
    z = ent_unl[class_name]
    
    fig = plt.figure(figsize=(20, 10))
    plt.hist(np.log(x), bins=100, alpha=0.5, density=True, color="tab:blue", label=class_name)
    plt.hist(np.log(z), bins=100, alpha=0.5, density=True, color="tab:orange", label="NO 15/9-1")
    plt.legend(fontsize=20)
    plt.xlabel("log Entropy", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"{class_name}_entropy_distribution.png", dpi=300)
    plt.close()


    z = e[y == i]
    z = np.log(z)

    a = np.linspace(0, 1, 100)

    b = np.zeros_like(a)

    for j, alpha in enumerate(a):
        q = np.quantile(x, 1 - alpha)
        b[j] = np.sum(z < q) / len(z)

    plt.figure(figsize=(20, 10))
    for f in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
        
        r = f / (1 - f)
        r = r * len(e) / len(z)
        fdr = 1 / (1 + (1 - a) * r / b)

        plt.plot(a, fdr, label=f"f={f} (r={np.round(r, 2)})", marker="o")
    plt.legend(fontsize=20)
    plt.xlabel("alpha", fontsize=20)
    plt.ylabel("False Discovery rate", fontsize=20)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"{lab_to_name[i]}_fdr.png", dpi=300)
    plt.close()










# import normal distribution
from scipy.stats import norm

plt.figure(figsize=(20, 10))

for r in [0.0, 0.1, 0.2, 0.5, 1.0, 5.0]:


    a = np.linspace(0.01, 0.99, 100)
    q = norm.ppf(1 - a, 0, 1)
    b = norm.cdf(q, 2, 1)

    x = 1 + (1 - a) * r / b
    x = 1 / x



    plt.plot(a, x, label=f"r={r}")
plt.legend(fontsize=20)
plt.xlabel(r"$\alpha$", fontsize=20)
plt.ylabel("FDR", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r"$\mu=1$", fontsize=20)
plt.savefig("fdr_vs_alpha_fixed_mu.png", dpi=300)
plt.close()


plt.figure(figsize=(20, 10))

for mu_diff in [0.0, 0.5, 1.0, 2.0, 3.0]:


    a = np.linspace(0.01, 0.99, 100)
    q = norm.ppf(1 - a, 0, 1)
    b = norm.cdf(q, 1 + mu_diff, 1)

    x = 1 + (1 - a) * 0.1 / b
    x = 1 / x



    plt.plot(a, x, label=r"$\mu_{out} - \mu_{in}=$" + f"{mu_diff}")
plt.legend(fontsize=20)
plt.xlabel(r"$\alpha$", fontsize=20)
plt.ylabel("FDR", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r"$r=0.1$", fontsize=20)
plt.savefig("fdr_vs_alpha_fixed_r.png", dpi=300)
plt.close()