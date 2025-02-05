import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))

import glob
import argparse
import h5py
import torch
import torchvision
import vit.vit_utils as vit_utils
import numpy as np
import vit.vision_transformer as vits
from torchvision import transforms as pth_transforms
from vit.hdf5_dataloader_v2 import HDF5Dataset

print(f"Using {torch.cuda.get_device_name(0)}.")
print(f"Using GPU with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB memory.")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./hdf5/6407_6-5 1200 mDC.hdf5")
parser.add_argument("--pretrained_weights", type=str, default='/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370959/checkpoint.pth')
parser.add_argument("--output_dir", type=str, default="./features")
args = parser.parse_args()

transform = pth_transforms.Compose([
    pth_transforms.Resize((256, 256), interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, img_size=[224])

vit_utils.load_pretrained_weights(model, args.pretrained_weights, "teacher", "vit_small", 16)
model.eval()
model.cuda()

file_paths = glob.glob(args.data_path + "/*.hdf5")
file_paths.sort()

os.makedirs(args.output_dir, exist_ok=True)

for path in file_paths:
    output_fname = os.path.basename(path).replace(".hdf5", "_features.hdf5")
    output_fname = os.path.join(args.output_dir, output_fname)
    if os.path.exists(output_fname):
        print(f"Skipping {path}")
        continue
    
    print(f"Processing {path}")
    
    ds = HDF5Dataset(path, transform=transform)
    #args.pretrained_weights = '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint.pth'
    #ds = torchvision.datasets.ImageFolder('/Users/ima029/Desktop/NO 6407-6-5/labelled imagefolders/imagefolder_20/', transform=transform)
    #output_fname = "./labelled_crops_features.hdf5"

    data_loader = torch.utils.data.DataLoader(
        ds, 
        batch_size=128, 
        shuffle=False,
        num_workers=10,)

    filenames = [f[0] for f in ds.samples]
    filenames = [str(f) for f in filenames]
    labels = [f[1] for f in ds.samples]
    labels = [str(f) for f in labels]

    features = []

    for i, (samples, _) in enumerate(data_loader):
        print(f"Batch {i+1}/{len(data_loader)}")
        samples = samples.cuda()
        features.append(model(samples).detach().cpu().numpy())

    features = np.concatenate(features, axis=0)

    with h5py.File(output_fname, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("filenames", data=filenames)
        f.create_dataset("labels", data=labels)

print("Done.")