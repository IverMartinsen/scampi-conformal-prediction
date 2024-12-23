import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ref_ent = pd.read_csv('./postprocessing/entropy_distribution.csv')
df = pd.read_csv('./postprocessing/entropies_15_9_1.csv')

e = df["entropy"].values
y = df["prediction"].values


norm_stats = pd.read_csv('/Users/ima029/Desktop/NO 6407-6-5/entropy_fold_0.csv', index_col=0)
# import normal distribution

from scipy.stats import norm



plt.hist(e, bins=100, alpha=0.5, density=True, color="tab:orange", label="NO 15/9-1")
plt.legend(fontsize=20)
plt.xlabel("log Entropy", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("entropy_distribution.png", dpi=300)
plt.close()



for i in [0, 4, 11, 14]:

    class_name = lab_to_name[i]
    mean, std = norm_stats.loc[["mean", "std"], class_name]
    x = norm.rvs(loc=mean, scale=std, size=10000)
    
    #x = ref_ent.iloc[:, i].values
    z = e[y == i]
    
    fig = plt.figure(figsize=(20, 10))
    plt.hist(x, bins=100, alpha=0.5, density=True, color="tab:blue", label=class_name)
    
    plt.hist(np.log(z), bins=100, alpha=0.5, density=True, color="tab:orange", label="NO 15/9-1")
    plt.legend(fontsize=20)
    plt.xlabel("log Entropy", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"{class_name}_entropy_distribution.png", dpi=300)
    plt.close()


bias = pd.read_csv('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/figures/bias_15_9_1.csv')

for i in [0, 4, 11, 14]:

    class_name = lab_to_name[i]
    #x = ref_ent.iloc[:, i].values
    #x = np.log(x)
    mean, std = norm_stats.loc[["mean", "std"], class_name]
    x = norm.rvs(loc=mean, scale=std, size=10000)



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