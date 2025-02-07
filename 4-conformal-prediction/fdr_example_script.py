# import normal distribution
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

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