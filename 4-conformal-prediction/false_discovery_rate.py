import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
args = parser.parse_args()

with open(os.path.join(args.src, 'ref_entropy.json'), 'r') as f:
    ent_lab = json.load(f)

with open(os.path.join(args.src, 'entropy.json'), 'r') as f:
    ent_unl = json.load(f)

with open(os.path.join(args.src, 'lab_to_name.json'), 'r') as f:
    lab_to_name = json.load(f)

def compute_fdr(alpha, beta, P, N):
    return 1 / (1 + ((1 - alpha) / beta) * (P / N))

def compute_T(ent_unl):
    return np.array([len(ent_unl[k]) for k in ent_unl.keys()]).sum()

def compute_P(T, gamma):
    return T * gamma

def compute_N(ent_unl, class_name, P):
    T_class = len(ent_unl[class_name])
    #assert T_class > P, print(T_class, P)
    return np.max((T_class - P, 0))

def compute_beta(x, z, alpha):
    """
    Given alpha, how many elements in z are smaller than the alpha-quantile of x
    """
    #q = np.quantile(x, 1 - alpha)
    q = norm.ppf(1 - alpha, loc=np.mean(x), scale=np.std(x))
    #p = np.sum(z < q) / len(z)
    p = norm.cdf(q, loc=np.mean(z), scale=np.std(z))
    return p

for class_name in ent_lab.keys():

    x = ent_lab[class_name]
    z = np.array(ent_unl[class_name])
    #z = np.concatenate([ent_unl[k] for k in ent_unl.keys() if k != class_name])
    #z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
    # remove nan values
    z = z[~np.isnan(z)]
    a = np.linspace(0, 0.99, 100) # alphas

    b = np.zeros_like(a) # betas

    for j, alpha in enumerate(a):
        b[j] = compute_beta(x, z, alpha)
        
    #if class_name == "inaperturopollenites":
    #    breakpoint()
    
    plt.figure(figsize=(16, 8))
    for gamma in [0.0001, 0.001, 0.01]:
        
        T = compute_T(ent_unl)
        #T = len(z)
        P = compute_P(T, gamma)
        N = compute_N(ent_unl, class_name, P)
        fdr = compute_fdr(a, b, P, N)

        plt.plot(a, fdr, label=r"$\gamma =$" + f"{gamma} " + r"($\frac{P}{N}=$" + f"{np.round(P / N, 2)})", marker="o")
    for a in [0.05, 0.50, 0.95]:
        plt.axvline(a, linestyle="--", color="black")
    plt.legend(fontsize=20)
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("FDR", fontsize=20)
    plt.ylim(0, 1)
    plt.xticks([0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00], fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.src, f"{class_name}_fdr.png"), dpi=300)
    plt.close()

# joint plots
for gamma in [1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01]:

    plt.figure(figsize=(14, 7))

    #for class_name in ['bisaccate']:
    #for class_name in ['alisocysta', 'bisaccate', 'inaperturopollenites', 'palaeoperidinium']:
    for class_name in ['dissiliodinium', 'rigaudella', 'sirmiodinium', 'surculosphaeridium']:
        x = ent_lab[class_name]
        z = ent_unl[class_name]
        z = np.array(z)
        z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
        # remove nan values
        z = z[~np.isnan(z)]
        a = np.linspace(0, 0.99, 100) # alphas

        b = np.zeros_like(a) # betas

        for j, alpha in enumerate(a):
            b[j] = compute_beta(x, z, alpha)        
            
        T = compute_T(ent_unl)
        P = compute_P(T, gamma)
        N = compute_N(ent_unl, class_name, P)
        fdr = compute_fdr(a, b, P, N)

        plt.plot(a, fdr, label=f"{class_name.capitalize()}", marker="o", linewidth=4)

    for a in [0.05, 0.50, 0.95]:
        plt.axvline(a, linestyle="--", color="black", linewidth=4)
    plt.legend(fontsize=30)
    plt.xlabel(r"$\alpha$", fontsize=30)
    plt.ylabel("FDR", fontsize=30)
    plt.ylim(0, 1)
    plt.xticks([0.05, 0.50, 0.95], fontsize=30, rotation=45)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(args.src, f"gamma_{gamma}_fdr.jpg"), dpi=300)
    plt.close()



# compute FDR for given alpha and gamma and class_name

#alpha = 0.95
#gamma = 0.0001
#class_name = "palaeoperidinium"
#
#x = ent_lab[class_name]
#z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
#
#beta = compute_beta(x, z, alpha)
#
#T = compute_T(ent_unl)
#P = compute_P(T, gamma)
#N = compute_N(ent_unl, class_name, P)
#fdr = compute_fdr(alpha, beta, P, N)
#
#print(f"False Discovery Rate for {class_name}: {fdr} (alpha={alpha}, gamma={gamma})")



