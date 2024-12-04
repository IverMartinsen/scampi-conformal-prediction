import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils import read_fn


def init_centroids_semi_supervised(x_lab, y_lab, x_un, k):
    """Init centroids semi supervised using kmeans++

    Args:
        x_lab (_type_): features of labeled data
        y_lab (_type_): labels of labeled data
        x_un (_type_): features of unlabeled data
        k (_type_): number of clusters
    """
    rng = np.random.default_rng(seed=42)
    
    k_lab = np.unique(y_lab).shape[0]
    k_un = k - k_lab

    cluster_labs = np.zeros_like(y_lab)

    centroids = np.zeros((k, x_lab.shape[1]))
    for i, lab in enumerate(np.unique(y_lab)):
    #for i, lab in tqdm(enumerate(np.unique(y_lab)), total=k_un):
        centroids[i] = np.mean(x_lab[y_lab == lab], axis=0)
        cluster_labs[y_lab == lab] = i

    # fill the rest of the centroids with the first centroid
    for i in range(k_lab, k):
        centroids[i] = centroids[0]
    
    for i in tqdm(range(k_un), total=k_un):
        d = euclidean_distances(x_un, centroids).min(axis=1)
        p = d / d.sum()
        centroids[k_lab + i] = x_un[rng.choice(range(x_un.shape[0]), p=p)]
    
    return centroids, cluster_labs


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        filenames = f["filenames"][()]
        features = f["features"][()]
        labels = f["labels"][()]

    filenames = np.array([f.decode() for f in filenames])
    labels = np.array([int(f.decode()) for f in labels])
    return filenames, features, labels


# load unlabelled features
path_to_features = '/Users/ima029/Desktop/NO 6407-6-5/features/6407_6-5 2020 mDC_features.hdf5'
path_to_images = "./hdf5/6407_6-5 2020 mDC.hdf5"

f_un, x_un, _ = load_hdf5(path_to_features)

import joblib

ood_detector = joblib.load("ood_detection_model.pkl")

y_pred = ood_detector.predict(x_un)

int_to_class = {
    0: "ok",
    1: "black",
    2: "dark",
    3: "blurry",
    4: "artifact",
    5: "multi",
}

for i in np.unique(y_pred):
    fnames = f_un[y_pred == i]
    # random sample
    fnames = np.random.choice(fnames, 100, replace=False)
    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for j, ax in enumerate(ax.flatten()):
        try:
            with h5py.File(path_to_images, 'r') as f:
                img = f[fnames[j]][()]
                img = read_fn(img)
                img = Image.fromarray(img)
                img = img.resize((224, 224))
            #img = Image.open(f"{path_to_images}/{filenames_[j]}")
            ax.imshow(img)


        except IndexError:
            pass
        ax.axis("off")
    fig.suptitle(f"Group {int_to_class[i]}", fontsize=20)
    plt.savefig(f"Group_{int_to_class[i]}.png")
    plt.close()

x_un = x_un[y_pred == 0]
f_un = f_un[y_pred == 0]

np.unique(y_pred, return_counts=True)


# load labelled features
path_to_labelled_features = "labelled_crops_features.hdf5"

lab_filenames, lab_features, lab_labels = load_hdf5(path_to_labelled_features)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=20)
model.fit(lab_features, lab_labels)

model.score(lab_features, lab_labels)

joblib.dump(model, "genus_classifier.pkl")


class_of_interest = 14

f_lab = np.array(lab_filenames)[np.where(lab_labels == class_of_interest)]
x_lab = lab_features[np.where(lab_labels == class_of_interest)]
y_lab = lab_labels[np.where(lab_labels == class_of_interest)]

ood_detector = joblib.load("ood_detection_model.pkl")
a = ood_detector.predict(lab_features)

b = lab_filenames[a == 5]

for file in b:
    print(file)

np.unique(a, return_counts=True)

# concatenate features
x_tot = np.concatenate([x_lab, x_un], axis=0)

# init centroids
centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 100)

kmeans = KMeans(n_clusters=100, random_state=0, init=centroids)
kmeans.fit(x_tot)

shifted_labels = kmeans.labels_[len(x_lab):]

filenames_ = f_un[shifted_labels == 0]

a = model.predict_proba(x_un[shifted_labels == 0])

a = model.predict_proba(lab_features)

# compute entropy
entropy = -np.sum(a * np.log(a), axis=1)
entropy.mean()

len(filenames_)

df = pd.DataFrame({"filename": filenames_, "label": [class_of_interest] * len(filenames_)})
df.to_csv(f"./feature_stats/{os.path.basename(path_to_features).replace('_features.hdf5', '.csv')}", index=False)

for file in filenames_:
    with h5py.File(path_to_images, 'r') as f:
        img = f[file][()]
        img = read_fn(img)
        img = Image.fromarray(img)
        os.makedirs("./extracted_crops/inaperturopollenites", exist_ok=True)
        img.save(f"./extracted_crops/inaperturopollenites/{file}.png")



for i in [0]:
    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for j, ax in enumerate(ax.flatten()):
        try:
            with h5py.File(path_to_images, 'r') as f:
                img = f[filenames_[j]][()]
                img = read_fn(img)
                img = Image.fromarray(img)
            #img = Image.open(f"{path_to_images}/{filenames_[j]}")
            ax.imshow(img)
        except IndexError:
            pass
        ax.axis("off")
    fig.suptitle(f"Group {i}", fontsize=20)
    plt.savefig(f"Group_{i}.png")
    plt.close()







kmeans.labels_[:50]





(euclidean_distances(x_lab) + np.eye(50) * 40).max()

D = euclidean_distances(x_lab, features)
nn = np.argsort(D, axis=1)[:, :1]

shifted_labels = kmeans.labels_[50:]

b = np.where(D[:, np.where(shifted_labels == 80)[0]])[1]

a = np.where(shifted_labels == 80)[0][b]

a = np.unique(a)

idx = a

idx = np.unique(np.where(D < 45)[1])

np.stack(shifted_labels, 50)

idx, counts = np.unique(nn, return_counts=True)
idx = idx[counts > 1]


for i in idx:
    fname = f_un[i]
    with h5py.File(path_to_images, 'r') as f:
        image = f[fname][()]
        image = read_fn(image)
        image = Image.fromarray(image)
    image.show()


plt.hist(D.flatten(), bins=100)
plt.savefig("hist.png")

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x_lab)
y_lab = kmeans.labels_



for i in range(1):
    fnames = f_lab[kmeans.labels_ == i]
    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for j, ax in enumerate(ax.flatten()):
        try:
            img = Image.open(fnames[j])
            ax.imshow(img)
        except IndexError:
            pass
        ax.axis("off")
    fig.suptitle(f"Group {i}", fontsize=20)
    plt.savefig(f"Group_{i}.png")
    plt.close()
    






kmeans.labels_[:697][np.where(y_lab == 11)]

kmeans.labels_[:50]


path_to_images = '/Users/ima029/Desktop/NO 6407-6-5/100K_BENCHMARK_224x224/images'
(kmeans.labels_ == 80).sum()


for i in range(100):
    x = (kmeans.labels_[50:] == i).sum()
    print(f"Group {i}: {x}")


# sort df by filename
df = df.sort_values("filename").reset_index(drop=True)

try:
    X = df.drop(columns=["label", "filename"])
except KeyError:
    X = df.drop(columns=["filename"])

F = df["filename"]

os.makedirs(destination, exist_ok=True)

known_sediment =  [7, 50, 365, 426, 447]
known_benthic = [54, 106, 331]
known_planktic = [57, 439, 460]
known = np.array(known_sediment + known_benthic + known_planktic)

x_lab = np.array(X.loc[known])
y_lab = np.array([0] * len(known_sediment) + [1] * len(known_benthic) + [2] * len(known_planktic))
x_un = np.array(X.drop(known))

centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 10)

kmeans = KMeans(n_clusters=10, random_state=0, init=centroids)

group_labels = kmeans.fit_predict(X)

for i in range(10):
    filenames = F[group_labels == i]
    features = X[group_labels == i]
    centroid = kmeans.cluster_centers_[i]
    mean_distance = np.mean(np.linalg.norm(features - centroid, axis=1))
    n = np.sqrt(len(filenames))
    n = int(n) + 1 if n % 1 != 0 else int(n)
    fig, ax = plt.subplots(n, n, figsize=(20, 20))
    for j, filename in enumerate(filenames):
        img = Image.open(f"{path_to_crops}/{filename}").resize((224, 224))
        ax.flatten()[j].imshow(img)
    for ax in ax.flatten():
        ax.axis("off")
    fig.suptitle(f"Group {i} - Mean Distance: {mean_distance:.2f}", fontsize=20)
    plt.savefig(f"{destination}/Group__with_knowns{i}.png")
    plt.close()
