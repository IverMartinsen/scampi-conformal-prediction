import os
import glob
import h5py
import time
import torch
import torchvision
import concurrent.futures
from PIL import Image


# list paths to all shards
folder = '/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/images/'
output_file = '/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/images.hdf5'

read_fn = lambda file: (file, torchvision.io.read_file(file))

print(f"Processing {folder}...")
files = glob.glob(os.path.join(folder, "*.jpg"))
print(f"Number of files: {len(files)}")

file_bytes = {}
# read files in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(read_fn, file) for file in files]
    for future in concurrent.futures.as_completed(futures):
        file, image = future.result()
        file_bytes[file] = image

with h5py.File(output_file, 'w') as f:
    for file, bytes in file_bytes.items():
        basename = os.path.basename(file)
        f.create_dataset(basename, data=bytes, dtype='uint8')

print(f"Finished processing {folder}!")



# test the hdf5 files
if __name__ == "__main__":
    def read_fn(bytes):
        image = torch.tensor(bytes) # sequence of bytes
        image = torchvision.io.decode_jpeg(image) # shape: (3, H, W)
        image = image.permute(1, 2, 0) # shape: (H, W, 3)
        return image.numpy()
    path_to_hdf5 = output_file
    # read the hdf5 file
    with h5py.File(path_to_hdf5, 'r') as f:
        keys = list(f.keys())

    for i, key in enumerate(keys):
        with h5py.File(path_to_hdf5, 'r') as f:
            image = f[key][()] # fancy indexing to get the value of the dataset
            image = read_fn(image)
            image = Image.fromarray(image)
            image.show()
            if i == 10:
                break
