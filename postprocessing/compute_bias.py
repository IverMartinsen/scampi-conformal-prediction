import os
import sys
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from postprocessing.utils import load_hdf5


lab_to_name = {
    0: "alisocysta",
    1: "areoligera",
    2: "areosphaeridium",
    3: "azolla",
    4: "bisaccate",
    5: "cleistosphaeridium",
    6: "deflandrea",
    7: "eatonicysta",
    8: "glaphyrocysta",
    9: "hystrichokolpoma",
    10: "hystrichosphaeridium",
    11: "inaperturopollenites",
    12: "isabelidinium",
    13: "palaeocystodinium",
    14: "palaeoperidinium",
    15: "phthanoperidinium",
    16: "spiniferites",
    17: "subtilisphaera",
    18: "svalbardella",
    19: "wetzeliella",
    }

path = './data/NO 15-9-1/features'
path_to_files = glob.glob(path + "/*.hdf5")
path_to_files.sort()

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

classifier = LinearClassifier(384, 20)
classifier.load_state_dict(torch.load('./postprocessing/trained_models/vit_small/classifier_20241216122634.pth', map_location="mps"))

predictions = []
entropies = []

for path_to_features in path_to_files:

    # =============================================================================
    # FEATURE LOADING STEP
    # =============================================================================
    f_un, x_un, _ = load_hdf5(path_to_features)
    print(f"Loaded {x_un.shape[0]} features.")

    # =============================================================================
    # OOD DETECTION STEP
    # =============================================================================
    ood_detector = joblib.load('./postprocessing/ood_detector/ood_detector.pkl')
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
    # PREDICTION STEP
    # =============================================================================
    logits = classifier(torch.tensor(x_un).float()).detach().numpy()
    y_prob = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    entropy = -np.sum(y_prob * np.log(y_prob), axis=1)
    entropies.append(entropy)
    y_pred = np.argmax(y_prob, axis=1)
    predictions.append(y_pred)

predictions = np.concatenate(predictions)
entropies = np.concatenate(entropies)

#save entropies
pd.DataFrame({"entropy": entropies, "prediction": predictions}).to_csv("entropies.csv", index=False)

classes, counts = np.unique(predictions, return_counts=True)

n = len(predictions)

bias = counts / n

pd.DataFrame({"class": [lab_to_name[lab] for lab in classes], "bias": bias}).to_csv("bias.csv", index=False)


c = [lab_to_name[lab] for lab in classes]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(classes, bias)
plt.xticks(classes, [lab_to_name[lab] for lab in classes], rotation=45, ha="right")
plt.axhline(y=1/20, color='r', linestyle='--')
plt.ylabel("Bias")
plt.title("Bias of the classifier")
plt.tight_layout()
plt.savefig("bias.png", dpi=300)
plt.close()