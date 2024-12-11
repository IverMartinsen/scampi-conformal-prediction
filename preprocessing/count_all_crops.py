import os
import sys

sys.path.append(".")

import glob
import h5py
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from vit.hdf5_dataloader_v2 import HDF5Dataset

parser = argparse.ArgumentParser(description='path')
parser.add_argument('--path', type=str, default="preprocessing/hdf5/", help='path to tfrecords')
parser.add_argument('--filetype', type=str, default="hdf5", help='filetype')
args = parser.parse_args()

if __name__ == "__main__":

    files = glob.glob(args.path + f"/*.{args.filetype}")
    files.sort()

    counts = {}

    for file in files:
        if args.filetype == "hdf5":
            ds = HDF5Dataset(file)
            counts[os.path.basename(file)] = len(ds)
        elif args.filetype == "tfrecords":
            ds = tf.data.TFRecordDataset(file)
            i = 0
            for _ in ds:
                i += 1
            counts[os.path.basename(file)] = i

    df = pd.DataFrame(counts.items(), columns=["filename", "count"])
    df.to_csv("counts.csv", index=False)
