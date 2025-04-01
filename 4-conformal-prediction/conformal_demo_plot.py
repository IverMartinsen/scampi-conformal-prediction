import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

q = 1.35

plt.figure(figsize=(10, 5))
x = np.linspace(0, 2, 1000)
plt.plot(x, norm.pdf(x, loc=1.50, scale=0.15), label='OOD data')
plt.plot(x, norm.pdf(x, loc=1.00, scale=0.2), label='ID data')
plt.axvline(q, color='black', linestyle='--', label='Rejection threshold')
plt.fill_between(x, norm.pdf(x, loc=1.50, scale=0.15), where=x < q, alpha=0.5, label=r'$\beta=0.15$')
plt.fill_between(x, norm.pdf(x, loc=1.00, scale=0.2), where=x > q, alpha=0.5, label=r'$\alpha=0.05$')
plt.xlabel('<- Entropy ->')
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values(): # remove frame
    spine.set_visible(False)
plt.legend()
plt.savefig('non_conformity_scores.jpg', dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
x = np.linspace(0, 2, 1000)
plt.plot(x, norm.pdf(x, loc=1.50, scale=0.15), label='OOD data')
plt.plot(x, norm.pdf(x, loc=1.00, scale=0.2), label='ID data')
plt.axvline(q, color='black', linestyle='--', label='Rejection threshold')
plt.fill_between(x, norm.pdf(x, loc=1.50, scale=0.15), where=x < q, alpha=0.5, label=r'$\beta=0.15$')
plt.fill_between(x, norm.pdf(x, loc=1.00, scale=0.2), where=x < q, alpha=0.5, label=r'$1-\alpha=0.95$')
plt.xlabel('<- Entropy ->')
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values(): # remove frame
    spine.set_visible(False)
plt.legend()
plt.savefig('non_conformity_scores_reverse.jpg', dpi=300)
plt.close()
