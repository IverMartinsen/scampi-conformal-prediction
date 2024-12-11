import os
import glob
import tensorflow as tf
from PIL import Image

path_to_tfrecords = "./tfrecords_filtered_100K_benchmark/"
destination = "./imagefolder224filtered100k/images/"

os.makedirs(destination, exist_ok=True)

def _tfrecord_map_function(x):
    """Parse a single image from a tfrecord file."""
    # Dict with key 'image' and value of type string
    x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
    # Tensor of type uint8
    x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
    return x

ds = tf.data.TFRecordDataset(glob.glob(path_to_tfrecords + "*.tfrecords"))
ds = ds.map(_tfrecord_map_function)

for i, x in enumerate(ds):
    Image.fromarray(x.numpy()).save(destination + f"{i}.png")