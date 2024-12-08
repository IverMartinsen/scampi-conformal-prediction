import os
import glob
import h5py
import numpy as np
import pandas as pd
from hdf5_dataloader_v2 import HDF5Dataset

path = "hdf5"

files = glob.glob(path + "/*.hdf5")
files.sort()

counts = {}

for file in files:
    ds = HDF5Dataset(file)
    counts[os.path.basename(file)] = len(ds)

df = pd.DataFrame(counts.items(), columns=["filename", "count"])
df.to_csv("counts.csv", index=False)