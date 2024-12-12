import os
import glob
import h5py
import time
import torch
import argparse
import torchvision
import tensorflow as tf


parser = argparse.ArgumentParser(description='path')
parser.add_argument('--path', type=str, default="", help='path to tfrecords')
parser.add_argument('--destination', type=str, default="", help='destination path')
args = parser.parse_args()

args.path = '/Users/ima029/Desktop/NO 6407-6-5/data/NO 15-9-1/tfrecords'
args.destination = '/Users/ima029/Desktop/NO 6407-6-5/data/NO 15-9-1/hdf5'


def _tfrecord_map_function(x):
    """Parse a single image from a tfrecord file."""
    # Dict with key 'image' and value of type string
    x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
    # Tensor of type uint8
    x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
    return x


files = glob.glob(args.path + "/*.tfrecords")
files.sort()

os.makedirs(args.destination, exist_ok=True)

for i, file in enumerate(files):

    print(f"Processing {file}...")
    start = time.time()

    output_file = os.path.join(args.destination, os.path.basename(file).replace(".tfrecords", ".hdf5"))

    ds = tf.data.TFRecordDataset(file)
    ds = ds.map(_tfrecord_map_function)
    with h5py.File(output_file, 'w') as out:
        for i, img in enumerate(ds):
            img = torch.tensor(img.numpy(), dtype=torch.uint8)
            img = img.permute(2, 0, 1)
            bytes = torchvision.io.encode_jpeg(img, quality=95)
            out.create_dataset("crop_" + str(i).zfill(6), data=bytes.numpy(), dtype='uint8')

    end = time.time()
    print(f"Finished processing {file}!")
    print(f"Time taken: {end - start}")
    