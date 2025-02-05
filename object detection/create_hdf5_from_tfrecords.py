import os
import sys

sys.path.append(os.path.join(os.getcwd(), "object detection"))

import glob
import h5py
import time
import argparse
from generator_from_tfrecords import tf_generator

parser = argparse.ArgumentParser(description='path')
parser.add_argument('--src', type=str, default="", help='path to tfrecords folder')
parser.add_argument('--dst', type=str, default="", help='path and filename of the output hdf5 file')
args = parser.parse_args()

files = glob.glob(args.src + "/*.tfrecords")
files.sort()

start = time.time()

image_generator = tf_generator(files)

with h5py.File(args.dst, 'w') as out:
    for i, bytes in enumerate(image_generator):
        out.create_dataset("crop_" + str(i).zfill(7), data=bytes, dtype='uint8')

end = time.time()
print(f"Finished processing {args.path}")
print(f"Time taken: {end - start} seconds")
