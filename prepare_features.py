import os
import argparse
import h5py
import torch
import torchvision
import vit_utils
import numpy as np
import vision_transformer as vits
from torchvision import transforms as pth_transforms
from hdf5_dataloader_v2 import HDF5Dataset

data_path = "100K_BENCHMARK_224x224.hdf5"
pretrained_weights = '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370959/checkpoint.pth'
output_fname = "100K_BENCHMARK_224x224_features.hdf5"



imagefolder = '/Users/ima029/Desktop/NO 6407-6-5/labelled imagefolders/imagefolder_20/'
output_fname = 'labelled_crops_features.hdf5'


transform = pth_transforms.Compose([
    pth_transforms.Resize((256, 256), interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, img_size=[224])

vit_utils.load_pretrained_weights(model, pretrained_weights, "teacher", "vit_small", 16)
model.eval()
model.cuda()

ds = HDF5Dataset(data_path, transform=transform)
ds = torchvision.datasets.ImageFolder('/Users/ima029/Desktop/NO 6407-6-5/labelled imagefolders/imagefolder_20/', transform=transform)

data_loader = torch.utils.data.DataLoader(
    ds, 
    batch_size=128, 
    shuffle=False,
    num_workers=10,)

filenames = [f[0] for f in ds.samples]
labels = [f[1] for f in ds.samples]


filenames = [str(f) for f in filenames]

features = []

for i, (samples, _) in enumerate(data_loader):
    print(f"Batch {i+1}/{len(data_loader)}")
    #samples = samples.cuda()
    features.append(model(samples).detach().cpu().numpy())
    #filenames.append(_)

features = np.concatenate(features, axis=0)

with h5py.File(output_fname, "w") as f:
    f.create_dataset("features", data=features)
    f.create_dataset("filenames", data=filenames[:len(features)])
    f.create_dataset("labels", data=labels[:len(features)])
