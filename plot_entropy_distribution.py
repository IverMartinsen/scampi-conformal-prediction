import joblib
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf5


def compute_reference_entropy(classifier, class_label):
    # load labelled features
    path_to_labelled_features = "labelled_crops_features.hdf5"

    _, x, y = load_hdf5(path_to_labelled_features)

    p = classifier.predict_proba(x)[y == class_label]
    e = -np.sum(p * np.log(p), axis=1)

    return e


path_to_features = '/Users/ima029/Desktop/NO 6407-6-5/features/6407_6-5 2030 mDC_features.hdf5'
path_to_images = path_to_features.replace("_features", "")

f_un, x_un, _ = load_hdf5(path_to_features)

ood_detector = joblib.load("ood_detection_model.pkl")

pred = ood_detector.predict(x_un)

f_un = f_un[pred == 0]
x_un = x_un[pred == 0]

classifier = joblib.load("genus_classifier.pkl")

y_prob = classifier.predict_proba(x_un)
y_pred = np.argmax(y_prob, axis=1)
entropy = -np.sum(y_prob * np.log(y_prob), axis=1)

path_to_labelled_features = "labelled_crops_features.hdf5"

_, x_lab, y_lab = load_hdf5(path_to_labelled_features)

y_prob_lab = classifier.predict_proba(x_lab)
entropy_lab = -np.sum(y_prob_lab * np.log(y_prob_lab), axis=1)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
xlim = (np.log(entropy).min(), np.log(entropy).max())

int_to_class = {
    0: "alisocysta",
    4: "bisaccate",
    11: "inaperturopollenites",
    14: "palaeoperidinium",
}

fontsize = 20

q = np.quantile(entropy, 0.05)

t = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
    t[i] = (e > q).mean()




alpha = t.max()

q_class_wise = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
    n = len(e)
    q_class_wise[i] = np.quantile(e, 1 - alpha)


fig, axes = plt.subplots(5, 1, figsize=(20, 20), sharex=True)

ax = axes.flatten()[0]
ax.hist(np.log(entropy), bins=100, density=True, alpha=0.5, color=colors[0], label="Unlabelled data from slide")
ax.axvline(np.log(q), color='r', linestyle='--', label="5% quantile", linewidth=2)
ax.set_xlim(xlim)
ax.set_yticks([])
ax.legend(fontsize=fontsize)
for j, i in enumerate([0, 4, 11, 14]):
    ax = axes.flatten()[j + 1]
    ax.hist(np.log(entropy_lab)[y_lab == i], bins=10, alpha=0.5, density=True, color=colors[j + 1], label=f"{int_to_class[i]}")
    ax.axvline(np.log(q_class_wise[i]), color='r', linestyle='--', linewidth=2, label="adjusted quantile")
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.legend(fontsize=fontsize)
plt.xlabel("log Entropy", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"entropy_distribution{i}.png", dpi=300)
plt.close()
