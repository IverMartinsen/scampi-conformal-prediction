import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))


import glob
import time
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
import vit.vit_utils as vit_utils
import vit.vision_transformer as vits
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from vit.hdf5_dataloader_v2 import HDF5Dataset
from torchvision import transforms as pth_transforms
from postprocessing.utils import init_centroids_semi_supervised, lab_to_name

print(f"Using {torch.cuda.get_device_name(0)}.")
print(f"Using GPU with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB memory.")

parser = argparse.ArgumentParser()
parser.add_argument("--src_data", type=str, default='/Users/ima029/Desktop/NO 6407-6-5/data/NO 15-9-1/hdf5')
parser.add_argument("--pretrained_weights", type=str, default='/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370959/checkpoint.pth')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--path_to_labeled_crops", type=str, default='/Users/ima029/Desktop/NO 6407-6-5/data/labelled imagefolders/imagefolder_20')
parser.add_argument("--output_dir", type=str, default="./postprocessing/results/15-9-1-clustering")
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

path_to_files = glob.glob(args.src_data + "/*.hdf5")
path_to_files.sort()

os.makedirs(args.output_dir, exist_ok=True)

pre_model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, img_size=[224])
vit_utils.load_pretrained_weights(pre_model, args.pretrained_weights, "teacher", "vit_small", 16)

transform = pth_transforms.Compose([
    pth_transforms.Resize((256, 256), interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
])

norm_layer = torch.nn.Sequential(
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)

model = torch.nn.Sequential(norm_layer, pre_model).to(args.device).eval()

ds_labeled = torchvision.datasets.ImageFolder(args.path_to_labeled_crops, transform=transform)
dataloader_labeled = torch.utils.data.DataLoader(
    ds_labeled,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

lab_features = np.zeros((len(ds_labeled), 384))
lab_labels = np.zeros(len(ds_labeled))

for i, (x, y) in enumerate(tqdm(dataloader_labeled)):
    x = x.to(args.device)
    y = y.to(args.device)
    with torch.no_grad():
        lab_features[i * args.batch_size:(i + 1) * args.batch_size] = model(x).detach().cpu().numpy()
        lab_labels[i * args.batch_size:(i + 1) * args.batch_size] = y.detach().cpu().numpy()

detections = []

for slide in path_to_files:
    
    print(f"Processing {slide}...")

    start = time.time()
    
    ds = HDF5Dataset(slide, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    #x_un = np.zeros((len(ds), 384))
    x_un = []
    for i, (x, _) in enumerate(tqdm(dataloader)):
        x = x.to(args.device)
        with torch.no_grad():
            #x_un[i * args.batch_size:(i + 1) * args.batch_size] = model(x).detach().cpu().numpy()
            x_un.append(model(x).detach().cpu().numpy())
    x_un = np.concatenate(x_un, axis=0)

    for lab in [0, 4, 11, 14]:

        os.makedirs(os.path.join(args.output_dir, lab_to_name[lab]), exist_ok=True)
        
        x_lab = lab_features[np.where(lab_labels == lab)]
        y_lab = lab_labels[np.where(lab_labels == lab)]

        # concatenate features
        x_tot = np.concatenate([x_lab, x_un], axis=0)

        # init centroids
        centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 100)

        kmeans = KMeans(n_clusters=100, random_state=0, init=centroids)
        kmeans.fit(x_tot)

        shifted_labels = kmeans.labels_[len(x_lab):]

        retrieved_items = [x for x, _ in ds.samples[np.where(shifted_labels == 0)]]
        # save crops
        for idx in np.where(shifted_labels == 0)[0]:
            img = ds[idx][0]
            img = torch.tensor(img).permute(1, 2, 0).detach().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            name = os.path.basename(slide).replace(".hdf5", "") + "_" + os.path.basename(ds.samples[idx][0])
            img.save(os.path.join(args.output_dir, lab_to_name[lab], name + ".png"))
        
        detections += (list(zip([os.path.basename(slide).replace(".hdf5", "")] * len(retrieved_items), retrieved_items, [lab] * len(retrieved_items))))

    end = time.time()
        
    print(f"Done in {end - start:.2f} seconds.")
        
df = pd.DataFrame(detections, columns=["source", "filename", "label"])
df.to_csv(os.path.join(args.output_dir, "stats.csv"), index=False)
