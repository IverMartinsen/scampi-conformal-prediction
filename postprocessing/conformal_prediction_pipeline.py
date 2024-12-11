import os
import h5py
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_hdf5, read_fn, LinearClassifier


def compute_reference_entropy(classifier, class_label):
    # load labelled features
    path_to_labelled_features = "labelled_crops_features.hdf5"

    _, x, y = load_hdf5(path_to_labelled_features)

    p = classifier.predict_proba(x)[y == class_label]
    e = -np.sum(p * np.log(p), axis=1)

    return e


lab_to_name = {
    0: "alisocysta",
    3: "azolla",
    4: "bisaccate",
    11: "inaperturopollenites",
    14: "palaeoperidinium",
    }


path_to_features = '/Users/ima029/Desktop/NO 6407-6-5/features/6407_6-5 2030 mDC_features.hdf5'
folder = "./test analysis for 2030 mDC"
os.makedirs(folder, exist_ok=True)
global_alpha = 0.05

# =============================================================================
# FEATURE LOADING STEP
# =============================================================================
path_to_images = os.path.join("hdf5", os.path.basename(path_to_features).replace("_features", ""))
f_un, x_un, _ = load_hdf5(path_to_features)
print(f"Loaded {x_un.shape[0]} features.")

# =============================================================================
# OOD DETECTION STEP
# =============================================================================
ood_detector = joblib.load("ood_detection_model.pkl")
pred = ood_detector.predict(x_un)
_, counts = np.unique(pred, return_counts=True)
num_black = counts[1]
try:
    num_blurry = counts[2]
except IndexError:
    num_blurry = 0
print(f"Detected {num_black} black and {num_blurry} blurry images.")
print(f"Removing {num_black + num_blurry} images.")
f_un = f_un[pred == 0]
x_un = x_un[pred == 0]

# =============================================================================
# ENTROPY FILTERING STEP
# =============================================================================
classifier = joblib.load("genus_classifier.pkl")
y_prob = classifier.predict_proba(x_un)

classifier = LinearClassifier(384, 20)
classifier.load_state_dict(torch.load("classifier.pth", map_location="mps"))
logits = classifier(torch.tensor(x_un).float()).detach().numpy()
y_prob = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

y_pred = np.argmax(y_prob, axis=1)
entropy = -np.sum(y_prob * np.log(y_prob), axis=1)

q = np.quantile(entropy, global_alpha)

f_un = f_un[entropy < q]
x_un = x_un[entropy < q]
y_pred = y_pred[entropy < q]

# =============================================================================
# CLASS-WISE QUANTILE COMPUTATION
# =============================================================================
ref_ent = pd.read_csv("entropy_distribution.csv")

t = np.zeros(20)

for i in [0, 4, 11, 14]:
    #e = compute_reference_entropy(classifier, i)
    e = ref_ent.iloc[:, i].values
    t[i] = (e > q).mean()

alpha = t.max()

q_class_wise = np.zeros(20)

for i in [0, 4, 11, 14]:
    #e = compute_reference_entropy(classifier, i)
    e = ref_ent.iloc[:, i].values
    q_class_wise[i] = np.quantile(e, 1 - alpha)

# =============================================================================
# DETECTION STEP
# =============================================================================

detections = []
for k in [0, 4, 11, 14]:
    fnames = f_un[(y_pred == k) & (entropy[entropy < q] < q_class_wise[k])]

    for file in fnames:
        with h5py.File(path_to_images, 'r') as f:
            img = f[file][()]
            img = read_fn(img)
            img = Image.fromarray(img)
            os.makedirs(os.path.join(folder, lab_to_name[k]), exist_ok=True)
            img.save(os.path.join(folder, lab_to_name[k], f"{file}.png"))

    detections += (list(zip(fnames, [k] * len(fnames))))

df = pd.DataFrame(detections, columns=["filename", "label"])
df.to_csv(os.path.join(folder, os.path.basename(path_to_features).replace("_features.hdf5", ".csv")), index=False)
pd.DataFrame({'global_alpha': [global_alpha], 'class_wise_alpha': [alpha]}).to_csv(os.path.join(folder, "alpha.csv"), index=False)

# =============================================================================
# PLOTTING STEP
# =============================================================================
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
#xlim = (np.log(entropy).min(), np.log(entropy).max())
xlim = (min(np.log(entropy).min(), np.log(ref_ent.values).min()), np.log(entropy).max())
fontsize = 20
int_to_class = {
    0: "alisocysta",
    4: "bisaccate",
    11: "inaperturopollenites",
    14: "palaeoperidinium",
}

fig, axes = plt.subplots(5, 1, figsize=(20, 20), sharex=True)

ax = axes.flatten()[0]
ax.hist(np.log(entropy), bins=100, density=True, alpha=0.5, color=colors[0], label="Unlabelled data from slide")
ax.axvline(np.log(q), color='r', linestyle='--', label="5% quantile", linewidth=2)
ax.set_xlim(xlim)
ax.set_yticks([])
ax.legend(fontsize=fontsize)
for j, i in enumerate([0, 4, 11, 14]):
    ax = axes.flatten()[j + 1]
    #ax.hist(np.log(compute_reference_entropy(classifier, i)), bins=10, alpha=0.5, density=True, color=colors[j + 1], label=f"{int_to_class[i]}")
    ax.hist(np.log(ref_ent).iloc[:, i], bins=10, alpha=0.5, density=True, color=colors[j + 1], label=f"{int_to_class[i]}")
    ax.axvline(np.log(q_class_wise[i]), color='r', linestyle='--', linewidth=2, label="adjusted quantile")
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.legend(fontsize=fontsize)
plt.xlabel("log Entropy", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig(os.path.join(folder, "entropy_distribution_pytorch_classifier.png"), dpi=300)
plt.close()





x = ref_ent.iloc[:, 14]
z = entropy

fig = plt.figure(figsize=(20, 10))
plt.hist(np.log(x), bins=20, alpha=0.5, density=True, color="tab:blue", label="Palaeoperidinium")
plt.hist(np.log(z), bins=100, alpha=0.5, density=True, color="tab:orange", label="Unlabelled data from slide")
plt.legend(fontsize=20)
plt.xlabel("log Entropy", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("palaeoperidinium_entropy_distribution.png", dpi=300)
plt.close()

a = np.linspace(0, 1, 100)

b = np.zeros_like(a)

for i, alpha in enumerate(a):
    q = np.quantile(x, 1 - alpha)
    b[i] = np.sum(z < q) / len(z)


r = 0.8

plt.figure(figsize=(20, 10))
for r in [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 0.8]:
    fdr = 1 / (1 + (1 - a) * r / b)

    plt.plot(a, fdr, label=f"r={r}", marker="o")
plt.legend(fontsize=20)
plt.xlabel("alpha", fontsize=20)
plt.ylabel("False Discovery rate", fontsize=20)
plt.ylim(0, 1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("palaeoperidinium_detection_rate.png", dpi=300)
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