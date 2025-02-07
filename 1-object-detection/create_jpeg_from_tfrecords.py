import os
import glob
import torch
import argparse
import torchvision
import tensorflow as tf
from PIL import Image
from generator_from_tfrecords import tf_generator

parser = argparse.ArgumentParser(description='path')
parser.add_argument('--src', type=str, default="", help='path to tfrecords folder')
parser.add_argument('--dst', type=str, default="", help='path and filename of the output hdf5 file')
args = parser.parse_args()

files = glob.glob(args.src + "/*.tfrecords")
files.sort()

image_generator = tf_generator(files)

os.makedirs(args.src, exist_ok=True)

for i, x in enumerate(image_generator):
    x = torch.tensor(x)
    x = torchvision.io.decode_jpeg(x)
    x = x.permute(1, 2, 0)
    Image.fromarray(x.numpy()).save(args.src + "/crop_" + str(i).zfill(7) + ".jpg", quality=95)