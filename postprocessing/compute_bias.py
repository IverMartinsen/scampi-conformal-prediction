import glob
import json
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from postprocessing.utils import load_hdf5, lab_to_name



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
classifier.load_state_dict(torch.load('/Users/ima029/Desktop/NO 6407-6-5/postprocessing/trained_models/20250103120412/classifier.pth', map_location="mps"))

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
d = {}

for i in range(20):
    d[lab_to_name[i]] = entropies[predictions == i].tolist()

with open("entropies_15_9_1.json", "w") as f:
    json.dump(d, f)

# compute class frequency
classes, counts = np.unique(predictions, return_counts=True)

n = len(predictions)

bias = counts / n

pd.DataFrame({"class": [lab_to_name[lab] for lab in classes], "freq": bias}).to_csv("class_freq.csv", index=False)

c = [lab_to_name[lab] for lab in classes]

plt.figure(figsize=(10, 5))
plt.bar(classes, bias)
plt.xticks(classes, [lab_to_name[lab] for lab in classes], rotation=45, ha="right")
plt.axhline(y=1/20, color='r', linestyle='--')
plt.ylabel("Frequency")
plt.title("Prior frequency of classes")
plt.tight_layout()
plt.savefig("class_freq_15_9_1.png", dpi=300)
plt.close()