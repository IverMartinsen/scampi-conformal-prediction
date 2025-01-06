import os
import glob
import h5py
import json
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from utils import load_hdf5, read_fn, lab_to_name, LinearClassifier

# args
src_data = './data/NO 6407-6-5'
alpha = 0.05
path_to_ref_ent = './postprocessing/trained_models/merged_entropies.json'
path_to_ood_detector = './postprocessing/ood_detector/ood_detector.pkl'
path_to_classifier = './postprocessing/trained_models/20250103120412/classifier.pth'
# args end

path_to_files = os.path.join(src_data, "features")
path_to_files = glob.glob(path_to_files + "/*.hdf5")
path_to_files.sort()

path_to_hdf5 = os.path.join(src_data, "hdf5")

folder = "./postprocessing/results/" + os.path.basename(src_data) + f"_alpha_{alpha}"

os.makedirs(folder, exist_ok=True)

ood_detector = joblib.load(path_to_ood_detector)

classifier = LinearClassifier(384, 20)
classifier.load_state_dict(torch.load(path_to_classifier, map_location="mps"))

with open(path_to_ref_ent) as f:
    ref_ent = json.load(f)

detections = []

for path_to_features in path_to_files:
    # =============================================================================
    # FEATURE LOADING STEP
    # =============================================================================
    path_to_images = os.path.join(path_to_hdf5, os.path.basename(path_to_features).replace("_features", ""))
    f_un, x_un, _ = load_hdf5(path_to_features)
    print(f"Loaded {x_un.shape[0]} features.")

    # =============================================================================
    # OOD DETECTION STEP
    # =============================================================================
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
    logits = classifier(torch.tensor(x_un).float()).detach().numpy()
    y_prob = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    y_pred = np.argmax(y_prob, axis=1)
    entropy = -np.sum(y_prob * np.log(y_prob), axis=1)

    # =============================================================================
    # CLASS-WISE QUANTILE COMPUTATION AND DETECTION STEP
    # =============================================================================
    for k in [0, 4, 11, 14]:
        e = ref_ent[lab_to_name[k]]
        q = np.quantile(e, 1 - alpha)
    
        fnames = f_un[(y_pred == k) & (entropy < q)]

        for file in fnames:
            with h5py.File(path_to_images, 'r') as f:
                img = f[file][()]
                img = read_fn(img)
                img = Image.fromarray(img)
                os.makedirs(os.path.join(folder, lab_to_name[k]), exist_ok=True)
                img.save(os.path.join(folder, lab_to_name[k], f"{file}.png"))
        
        detections += (list(zip([os.path.basename(path_to_images)] * len(fnames), fnames, [k] * len(fnames))))

df = pd.DataFrame(detections, columns=["source", "filename", "label"])
df.to_csv(os.path.join(folder, "stats.csv"), index=False)
