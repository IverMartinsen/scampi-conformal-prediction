import os
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_hdf5, read_fn


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
y_pred = np.argmax(y_prob, axis=1)
entropy = -np.sum(y_prob * np.log(y_prob), axis=1)

q = np.quantile(entropy, global_alpha)

f_un = f_un[entropy < q]
x_un = x_un[entropy < q]
y_pred = y_pred[entropy < q]

# =============================================================================
# CLASS-WISE QUANTILE COMPUTATION
# =============================================================================
t = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
    t[i] = (e > q).mean()

alpha = t.max()

q_class_wise = np.zeros(20)

for i in [0, 4, 11, 14]:
    e = compute_reference_entropy(classifier, i)
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
