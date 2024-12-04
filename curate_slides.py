import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from utils import read_fn


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        filenames = f["filenames"][()]
        features = f["features"][()]
        labels = f["labels"][()]

    filenames = np.array([f.decode() for f in filenames])
    labels = np.array([int(f.decode()) for f in labels])
    return filenames, features, labels


path = "features"
files = glob.glob(path + "/*.hdf5")

sample = []

# list all files in all slides
for file in tqdm(files):
    filenames, _, _ = load_hdf5(file)
    for filename in filenames:
        sample.append((os.path.basename(file), filename))
    #sample.append(os.path.basename(file))
    #with h5py.File(file, 'r') as f:
    #    keys = list(f.keys())
    #    sample[os.path.basename(file)] = keys

sample = np.array(sample)

rng = np.random.default_rng(seed=42)
random_sample = sample[rng.choice(range(len(sample)), size=10000, replace=False)]

#random_sample[:, 0]

#features = np.zeros((len(random_sample), 384))
#
#for i, (slide, crop) in tqdm(enumerate(random_sample), total=len(random_sample)):
#    with h5py.File(os.path.join(path, slide), 'r') as f:
#        x = f["filenames"][()]
#        x = np.array([f.decode() for f in x])
#        y = f["features"][()]
#    features[i] = y[x == crop]


slides = []
crops = []
features = []

for slide in tqdm(files):
    x, y, z = load_hdf5(slide)
    subsample = random_sample[np.where(random_sample[:, 0] == os.path.basename(slide))][:, 1]
    idx = np.where(np.isin(x, subsample))[0]
    slides.append([os.path.basename(slide)] * len(idx))
    crops.append(x[idx])
    features.append(y[idx])

slides = np.concatenate(slides, axis=0)
crops = np.concatenate(crops, axis=0)
features = np.concatenate(features, axis=0)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(features)

for i in range(100):
    s = slides[kmeans.labels_ == i]
    path_to_images = [os.path.join("hdf5", x.replace('_features.hdf5', '.hdf5')) for x in s]
    fnames = crops[kmeans.labels_ == i]
    
    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for j, ax in enumerate(ax.flatten()):
        try:
            with h5py.File(path_to_images[j], 'r') as f:
                img = f[fnames[j]][()]
                img = read_fn(img)
                img = Image.fromarray(img)
                ax.imshow(img)
                ax.axis('off')
        except IndexError:
            pass

    plt.savefig(f"cluster_{i}.png")


class_to_int = {
    "ok": 0,
    "black": 1,
    "dark": 2,
    "blurry": 3,
    "artifact": 4,
    "multi": 5,
    }


black = [0, 8, 12, 34, 41, 56, 69]
dark = [23, 29, 33, 46, 48]
blurry = [14, 86, 94]
artifact = [6, 11, 25, 40, 52, 53, 89]
multi = [9, 17, 18, 37, 38, 39, 47, 60, 74, 78, 90, 95]

labels = np.zeros(len(features))


for i in range(100):
    if i in black:
        labels[kmeans.labels_ == i] = class_to_int["black"]
    elif i in dark:
        labels[kmeans.labels_ == i] = class_to_int["ok"]
    elif i in multi:
        labels[kmeans.labels_ == i] = class_to_int["ok"]
    elif i in blurry:
        labels[kmeans.labels_ == i] = class_to_int["blurry"]
    elif i in artifact:
        labels[kmeans.labels_ == i] = class_to_int["ok"]
    else:
        labels[kmeans.labels_ == i] = class_to_int["ok"]


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

model = LogisticRegression(random_state=0, max_iter=1000)
model = KNeighborsClassifier(n_neighbors=3)

model.fit(features, labels)

model.score(features, labels)

import joblib

joblib.dump(model, "ood_detection_model.pkl")