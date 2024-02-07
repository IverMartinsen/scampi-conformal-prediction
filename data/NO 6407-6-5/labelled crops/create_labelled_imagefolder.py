import os
import sys
import argparse
import shutil

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from scampi_evaluation.supervised_utils import stratified_idxs


parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=None)
parser.add_argument("--image_folder", type=str, default="./data/labelled crops")
parser.add_argument("--dest_folder", type=str, default="")
args = parser.parse_args()


if __name__ == "__main__":
    # Load metadata
    filenames_and_labels = pd.read_csv(os.path.join(args.image_folder, "labels.csv"))
    classnames_and_counts = pd.read_csv(os.path.join(args.image_folder, "label_count.csv"))
    
    labels = np.array(filenames_and_labels["label"])
    filenames = np.array(filenames_and_labels["filename"])
    classnames = np.array(classnames_and_counts["label"])
    # reduce number of classes
    idx_to_keep = np.where(labels < args.num_classes)[0]
    
    labels = labels[idx_to_keep]
    filenames = filenames[idx_to_keep]
    classnames = classnames[:args.num_classes]
    
    os.makedirs(os.path.join(args.dest_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.dest_folder, "val"), exist_ok=True)

    for folder in ["train", "val"]:
        for class_name in classnames:
            os.makedirs(os.path.join(args.dest_folder, folder, class_name), exist_ok=True)

    idx_train, idx_test = stratified_idxs(labels, splits=(0.8, 0.2), seed=42)

    for index in idx_train:
        path_to_file = os.path.join(args.image_folder, filenames[index])
        class_name = classnames[labels[index]]
        # copy file to train folder
        shutil.copy(path_to_file, os.path.join(args.dest_folder, "train", class_name))

    for index in idx_test:
        path_to_file = os.path.join(args.image_folder, filenames[index])
        class_name = classnames[labels[index]]
        # copy file to test folder
        shutil.copy(path_to_file, os.path.join(args.dest_folder, "val", class_name))
