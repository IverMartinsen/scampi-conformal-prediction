import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

with open('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/trained_models/merged_entropies.json', 'r') as f:
    ent_lab = json.load(f)

with open('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/trained_models/20250103120412/entropies_15_9_1.json', 'r') as f:
    ent_unl = json.load(f)

classes_of_interest = ['alisocysta', 'bisaccate', 'inaperturopollenites', 'palaeoperidinium']


for class_name in classes_of_interest:

    x = ent_lab[class_name]
    #z = ent_unl[class_name]
    z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
    
    fig = plt.figure(figsize=(20, 10))
    plt.hist(np.log(x), bins=20, alpha=0.5, density=True, color="tab:blue", label=class_name)
    plt.hist(np.log(z), bins=100, alpha=0.5, density=True, color="tab:orange", label="NO 15/9-1")
    plt.legend(fontsize=20)
    plt.xlabel("log Entropy", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"{class_name}_entropy_distribution.png", dpi=300)
    plt.close()


    recall = 1.00

    a = np.linspace(0, 0.99, 100)

    b = np.zeros_like(a)

    for j, alpha in enumerate(a):
        q = np.quantile(x, 1 - alpha)
        b[j] = np.sum(z < q) / len(z)

    plt.figure(figsize=(20, 10))
    for gamma in [0.001, 0.005, 0.01]:
        
        T = np.array([len(ent_unl[k]) for k in ent_unl.keys()]).sum()
        P = T * recall * gamma
        #T_star = len(z)
        T_star = len(ent_unl[class_name])
        assert T_star > P, print(T_star, P)
        N = T_star - P
        
        r = P / N
        
        fdr = 1 / (1 + (1 - a) * r / b)

        plt.plot(a, fdr, label=f"gamma={gamma} (r={np.round(r, 2)})", marker="o")
    plt.legend(fontsize=20)
    plt.xlabel("alpha", fontsize=20)
    plt.ylabel("False Discovery rate", fontsize=20)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"{class_name}_fdr.png", dpi=300)
    plt.close()


# compute FDR for given alpha and gamma and class_name

alpha = 0.95
gamma = 0.005
class_name = "inaperturopollenites"

x = ent_lab[class_name]
#z = ent_unl[class_name]
z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
q = np.quantile(x, 1 - alpha)
beta = np.sum(z < q) / len(z)

T = np.array([len(ent_unl[k]) for k in ent_unl.keys()]).sum()
P = T * recall * gamma
#T_star = len(z)
T_star = len(ent_unl[class_name])
N = T_star - P

fdr = 1 / (1 + (1 - alpha) * (P / N) / beta)
print(f"False Discovery Rate for {class_name}: {fdr} (alpha={alpha}, gamma={gamma})")
