import h5py
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

def read_fn(bytes):
    image = torch.tensor(bytes) # sequence of bytes
    image = torchvision.io.decode_jpeg(image) # shape: (3, H, W)
    image = image.permute(1, 2, 0) # shape: (H, W, 3)
    return image.numpy()

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        filenames = f["filenames"][()]
        features = f["features"][()]
        labels = f["labels"][()]

    filenames = np.array([f.decode() for f in filenames])
    labels = np.array([int(f.decode()) for f in labels])
    return filenames, features, labels

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
