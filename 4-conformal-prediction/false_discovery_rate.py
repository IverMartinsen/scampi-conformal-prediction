import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
args = parser.parse_args()

args.src = '/Users/ima029/Desktop/NO 6407-6-5/4-conformal-prediction/results/15-9-1 alpha 0.5'

with open(os.path.join(args.src, 'ref_entropy.json'), 'r') as f:
    ent_lab = json.load(f)

with open(os.path.join(args.src, 'entropy.json'), 'r') as f:
    ent_unl = json.load(f)

with open(os.path.join(args.src, 'lab_to_name.json'), 'r') as f:
    lab_to_name = json.load(f)

def compute_fdr(alpha, beta, P, N):
    return 1 / (1 + (1 - alpha) * P / N / beta)

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
    q = np.quantile(x, 1 - alpha)
    return np.sum(z < q) / len(z)

for class_name in ent_lab.keys():

    x = ent_lab[class_name]
    z = ent_unl[class_name]
    z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])
    
    a = np.linspace(0, 0.99, 100) # alphas

    b = np.zeros_like(a) # betas

    for j, alpha in enumerate(a):
        b[j] = compute_beta(x, z, alpha)

    plt.figure(figsize=(16, 8))
    for gamma in [0.001, 0.005, 0.01]:
        
        T = compute_T(ent_unl)
        P = compute_P(T, gamma)
        N = compute_N(ent_unl, class_name, P)
        fdr = compute_fdr(a, b, P, N)

        plt.plot(a, fdr, label=r"$\gamma =$" + f"{gamma} " + r"($\frac{P}{N}=$" + f"{np.round(P / N, 2)})", marker="o")
    plt.legend(fontsize=20)
    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("FDR", fontsize=20)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(args.src, f"{class_name}_fdr.png"), dpi=300)
    plt.close()


# compute FDR for given alpha and gamma and class_name

alpha = 0.95
gamma = 0.005
class_name = "inaperturopollenites"

x = ent_lab[class_name]
z = np.concatenate([ent_unl[k] for k in ent_unl.keys()])

beta = compute_beta(x, z, alpha)

T = compute_T(ent_unl)
P = compute_P(T, gamma)
N = compute_N(ent_unl, class_name, P)
fdr = compute_fdr(alpha, beta, P, N)

print(f"False Discovery Rate for {class_name}: {fdr} (alpha={alpha}, gamma={gamma})")



