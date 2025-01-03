import os
import glob
import h5py
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils import read_fn, init_centroids_semi_supervised, load_hdf5, lab_to_name

ood_detector = joblib.load("ood_detection_model.pkl")

# load labelled features
path_to_labelled_features = "labelled_crops_features.hdf5"

lab_filenames, lab_features, lab_labels = load_hdf5(path_to_labelled_features)

path_to_files = glob.glob("features" + "/*.hdf5")
path_to_files.sort()

for hdf5_file in path_to_files:

    print(f"Processing {hdf5_file}")
    
    # load unlabelled features
    path_to_features = hdf5_file
    path_to_images = os.path.join("./hdf5", os.path.basename(hdf5_file).replace('_features.hdf5', '.hdf5'))

    f_un, x_un, _ = load_hdf5(path_to_features)
    
    y_pred = ood_detector.predict(x_un)
    
    f_un = f_un[y_pred == 0]
    x_un = x_un[y_pred == 0]

    detections = []

    for lab in [3, 11, 14]:

        f_lab = np.array(lab_filenames)[np.where(lab_labels == lab)]
        x_lab = lab_features[np.where(lab_labels == lab)]
        y_lab = lab_labels[np.where(lab_labels == lab)]

        # concatenate features
        x_tot = np.concatenate([x_lab, x_un], axis=0)

        # init centroids
        centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 100)

        kmeans = KMeans(n_clusters=100, random_state=0, init=centroids)
        kmeans.fit(x_tot)

        shifted_labels = kmeans.labels_[len(x_lab):]

        filenames_ = f_un[shifted_labels == 0]

        detections += (list(zip(filenames_, [lab] * len(filenames_))))

        for file in filenames_:
            with h5py.File(path_to_images, 'r') as f:
                img = f[file][()]
                img = read_fn(img)
                img = Image.fromarray(img)
                os.makedirs(f"./extracted_crops/{lab_to_name[lab]}", exist_ok=True)
                img.save(f"./extracted_crops/{lab_to_name[lab]}/{file}.png")

    df = pd.DataFrame(detections, columns=["filename", "label"])
    df.to_csv(f"./feature_stats/{os.path.basename(path_to_features).replace('_features.hdf5', '.csv')}", index=False)
