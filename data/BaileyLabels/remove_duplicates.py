import os
import glob
import numpy as np
from PIL import Image

path = 'data/BaileyLabels/BaileyLabelSet'

files = glob.glob(path + '/*.jpg')
files = [os.path.basename(file) for file in files]


images = [Image.open(os.path.join(path, file)) for file in files]
images = [np.array(image) for image in images]

# remove duplicates

duplicate_indices = []

for i, image in enumerate(images):
    for j in range(i+1, len(images)):
        if image.shape != images[j].shape:
            continue
        if np.all(image == images[j]):
            print(("Found duplicate"))
            duplicate_indices.append(j)

print(f"Found {len(duplicate_indices)} duplicates")
duplicate_indices = np.unique(duplicate_indices)
print(f"Found {len(duplicate_indices)} unique duplicates")

for i in duplicate_indices:
    os.remove(os.path.join(path, files[i]))
    print(f"Removed {files[i]}")
