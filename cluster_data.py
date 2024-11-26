import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
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
    k_lab = np.unique(y_lab).shape[0]
    k_un = k - k_lab

    cluster_labs = np.zeros_like(y_lab)

    centroids = np.zeros((k, x_lab.shape[1]))
    for i, lab in enumerate(np.unique(y_lab)):
    #for i, lab in tqdm(enumerate(np.unique(y_lab)), total=k_un):
        centroids[i] = np.mean(x_lab[y_lab == lab], axis=0)
        cluster_labs[y_lab == lab] = i

    
    for i in tqdm(range(k_un), total=k_un):
        d = euclidean_distances(x_un, centroids).min(axis=1)
        p = d / d.sum()
        centroids[k_lab + i] = x_un[np.random.choice(range(x_un.shape[0]), p=p)]
    
    return centroids, cluster_labs


path_to_features = "100K_BENCHMARK_224x224_features.hdf5"
path_to_images = "100K_BENCHMARK_224x224.hdf5"

with h5py.File(path_to_features, 'r') as f:
    filenames = f["filenames"][()]
    features = f["features"][()]

filenames = [f.decode() for f in filenames]

with h5py.File(path_to_images, 'r') as f:
    keys = list(f.keys())

#images = []
#
#for i, key in enumerate(keys):
#    if key in filenames:
#        with h5py.File(path_to_images, 'r') as f:
#            image = f[key][()] # fancy indexing to get the value of the dataset
#            image = read_fn(image)
#            image = Image.fromarray(image)
#        images.append(image)


kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0)
kmeans.fit(features)
kmeans.labels_


path_to_labelled_features = "labelled_crops_features.hdf5"
with h5py.File(path_to_labelled_features, 'r') as f:
    lab_features = f["features"][()]
    lab_labels = f["labels"][()]
    lab_filenames = f["filenames"][()]

lab_filenames = [f.decode() for f in lab_filenames]

x_lab = lab_features[np.where(lab_labels == 11)]
y_lab = lab_labels[np.where(lab_labels == 11)]
y_lab = lab_labels
y_lab = np.arange(50)
f_lab = np.array(lab_filenames)[np.where(lab_labels == 11)]

random_idx = np.random.choice(range(features.shape[0]), 100)
x_un = features[random_idx]
f_un = np.array(filenames)[random_idx]

centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 100)



x_tot = np.concatenate([x_lab, x_un], axis=0)


#kmeans = MiniBatchKMeans(n_clusters=100, random_state=0, init=centroids)
kmeans = KMeans(n_clusters=100, random_state=1, init=centroids)
kmeans.fit(np.concatenate([x_lab, x_un], axis=0))
kmeans.labels_.shape


# update labels
kmeans.labels_[:50] = cluster_labels
# update centroids
for i in range(100):
    kmeans.cluster_centers_[i] = np.mean(x_tot[kmeans.labels_ == i], axis=0)
# update labels
kmeans.labels_ = kmeans.predict(x_tot)






centroids = kmeans.cluster_centers_
centroids[0] = np.mean(x_lab, axis=0)



D = euclidean_distances(x_lab, features)
nn = np.argmin(D, axis=1)

idx = np.unique(nn)

for i in nn:
    plt.imshow(Image.open(f"{path_to_images}/{filenames[i]}"))
    plt.show()
    plt.close()



kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x_lab)
y_lab = kmeans.labels_



for i in range(5):
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


for i in [0, 1, 42, 97]:
    filenames_ = f_un[kmeans.labels_[50:] == i]
    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for j, ax in enumerate(ax.flatten()):
        try:
            img = Image.open(f"{path_to_images}/{filenames_[j]}")
            ax.imshow(img)
        except IndexError:
            pass
        ax.axis("off")
    fig.suptitle(f"Group {i}", fontsize=20)
    plt.savefig(f"Group_{i}.png")
    plt.close()

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
