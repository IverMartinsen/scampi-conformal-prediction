import numpy as np
import tensorflow as tf

def _tfrecord_map_function(x):
    """Parse a single image from a tfrecord file."""
    # Dict with key 'image' and value of type string
    x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
    # Tensor of type uint8
    x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
    # Tensor of type string
    x = tf.image.encode_jpeg(x, quality=95)
    return x


# create custom iterator
class CustomGenerator:
    """
    Custom generator to convert tfrecord dataset to numpy array.
    Also converts the image from string to uint8.
    """
    def __init__(self, ds):
        self.ds = ds
        self.iterator = iter(self.ds)

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.iterator)
        x = np.frombuffer(x.numpy(), dtype=np.uint8)

        return x


def tf_generator(files):
    """
    Create a generator for tfrecords.
    Generates jpeg encoded images as numpy arrays of type uint8.
    """
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(_tfrecord_map_function)
    ds = CustomGenerator(ds)
    return ds
