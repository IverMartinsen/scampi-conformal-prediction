import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))

import json
import torch
import argparse
import torchvision
import numpy as np
import vit.vit_utils as vit_utils
import vit.vision_transformer as vits
from PIL import Image
from time import time
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--pretrained_weights", type=str, default="")
parser.add_argument("--data_dir", type=str, default="/Users/ima029/Desktop/NO 6407-6-5/data/labelled forams/imagefolders/merged")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--backbone_arch", type=str, default="vit_small")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation_split", type=float, default=None)
args = parser.parse_args()


from torchvision.transforms import v2

# load the data
print("Loading data...")
transform = transforms.Compose([
    #transforms.Resize((48, 48), interpolation=3),
    #v2.RandomResize(48, 224, interpolation=3),
    transforms.Resize((224, 224), interpolation=3),
    #transforms.RandomResizedCrop(224, scale=(0.5, 2.0), ratio = (1.0, 1.0), interpolation=Image.BICUBIC),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
])


dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
dataset_clean = torchvision.datasets.ImageFolder(args.data_dir)

for i, (img, _) in enumerate(dataset):
    img = img.permute(1, 2, 0).numpy()
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.show()
    if i == 10:
        break

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

print(f"Number of images: {len(dataset)}")

# load the model
pretrained_model = vits.__dict__[args.backbone_arch](patch_size=16, num_classes=0, img_size=[224])
vit_utils.load_pretrained_weights(pretrained_model, args.pretrained_weights, "teacher", args.backbone_arch, 16)

input_dim = {"vit_small": 384, "vit_base": 768}[args.backbone_arch]
output_dim = len(dataset.classes)

# extract the features
pretrained_model.eval()
pretrained_model.to(args.device)

x = np.zeros((len(dataset), input_dim))
y = np.zeros(len(dataset))
# iterate over the dataset four times to get the features


for i, (images, labels) in enumerate(dataloader):
    images = images.to(args.device)
    features = pretrained_model(images)
    x[i*args.batch_size:(i+1)*args.batch_size] = features.detach().cpu().numpy()
    y[i*args.batch_size:(i+1)*args.batch_size] = labels.detach().cpu().numpy()
    print(f"Batch {i+1}/{len(dataloader)}")

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=9)


log_model = LogisticRegression(multi_class="multinomial", class_weight="balanced", max_iter=1000, random_state=args.seed)

i_tr, i_val = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=y, random_state=args.seed)

#knn_model.fit(x[i_tr], y[i_tr])
knn_model.fit(x, y)

#knn_model.score(x[i_val], y[i_val])
knn_model.score(x, y)

#log_model.fit(x[i_tr], y[i_tr])
log_model.fit(x, y)
#log_model.score(x[i_val], y[i_val])
log_model.score(x, y)

from vit.hdf5_dataloader_v2 import HDF5Dataset

test_transform = transforms.Compose([
    transforms.Resize((48, 48), interpolation=3),
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])




#test_dataset = HDF5Dataset('/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/hdf5/images.hdf5', transform=test_transform)
test_dataset = torchvision.datasets.ImageFolder('/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/imagefolder', transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

x_test = np.zeros((len(test_dataset), input_dim))
for i, (images, _) in enumerate(test_loader):
    images = images.to(args.device)
    features = pretrained_model(images)
    x_test[i*args.batch_size:(i+1)*args.batch_size] = features.detach().cpu().numpy()
    print(f"Batch {i+1}/{len(test_loader)}")

y_pred = log_model.predict(x_test)
y_pred = knn_model.predict(x_test)
y_pred_proba = log_model.predict_proba(x_test)

#test_dataset_clean = HDF5Dataset('/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/hdf5/images.hdf5')
test_dataset_clean = torchvision.datasets.ImageFolder('/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/imagefolder')

output_dir = "eval_on_forams_35_imagefolder_knn_with_sediment"


y_pred = knn_model.predict(x_test)

for i in range(11):
    idx = np.where(y_pred == i)[0]
    images = [test_dataset_clean[i] for i in idx]
    os.makedirs(f"{output_dir}/{i}", exist_ok=True)
    for j, (img, _) in enumerate(images):
        img.save(f"{output_dir}/{i}/{j}.png")

# compute mean resolution for the test dataset

hs = []
ws = []
rgb_means = []
brightness = []
contrast = []
h = []
s = []
for i, (img, _) in enumerate(test_dataset_clean):
    hs.append(img.size[1])
    ws.append(img.size[0])
    rgb_means.append(np.mean(np.array(img), axis=(0, 1)))
    img = ImageEnhance.Brightness(img).enhance(100/69)
    brightness.append(np.mean(np.array(img)))
    contrast.append(np.std(np.array(img).mean(axis=2)))
    img = img.convert('HSV')
    h.append(np.mean(np.array(img)[:, :, 0]))
    s.append(np.mean(np.array(img)[:, :, 1]))
print(f"Mean resolution: {np.min(hs)}x{np.min(ws)}")
print(f"Mean RGB mean: {np.mean(rgb_means, axis=0)}")
print(f"Mean brightness: {np.mean(brightness)}")
print(f"Mean contrast: {np.mean(contrast)}")
print(f"Mean hue: {np.mean(h)}")
print(f"Mean saturation: {np.mean(s)}")




# normalize the brightness of the images
from PIL import ImageEnhance







path = '/Users/ima029/Desktop/NO 6407-6-5/data/labelled forams/imagefolders/merged/'
dest = '/Users/ima029/Desktop/NO 6407-6-5/data/labelled forams/imagefolders/merged_standardized'

import glob

for c in np.sort(os.listdir(path)):

    if c == '.DS_Store':
        continue
    
    images = glob.glob(os.path.join(path, c) + '/*.jpg')
    images = [Image.open(img) for img in images]

    hs = [img.size[1] for img in images]
    ws = [img.size[0] for img in images]
    qh = np.quantile(hs, 0.025)
    qw = np.quantile(ws, 0.025)
    rgb = [np.mean(np.array(img), axis=(0, 1)) for img in images]
    b = [np.mean(np.array(img)) for img in images]
    contrast = [np.std(np.array(img).mean(axis=2)) for img in images]
    images = [img.convert('HSV') for img in images]
    h = [np.mean(np.array(img)[:, :, 0]) for img in images]
    s = [np.mean(np.array(img)[:, :, 1]) for img in images]
    
    
    print(f"Class {c}: {qh}x{qw}")
    print(f"RGB mean: {np.mean(rgb, axis=0)}")
    print(f"Brightness: {np.mean(b)}")
    print(f"Contrast: {np.mean(contrast)}")
    print(f"Saturation: {np.mean(s)}")
    print(f"Hue: {np.mean(h)}")
    print("=====")


for c in np.sort(os.listdir(path)):

    if c == '.DS_Store':
        continue
    
    images = glob.glob(os.path.join(path, c) + '/*.jpg')
    basenames = [os.path.basename(img) for img in images]
    images = [Image.open(img) for img in images]

    brightness = np.mean([np.mean(np.array(img)) for img in images])
    contrast = np.mean([np.std(np.array(img).mean(axis=2)) for img in images])
    satu
    images = [ImageEnhance.Brightness(img).enhance(90/brightness) for img in images]
    images = [ImageEnhance.Contrast(img).enhance(90/contrast) for img in images]
    os.makedirs(os.path.join(dest, c), exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(dest, c, basenames[i]))
    






from postprocessing.utils import init_centroids_semi_supervised
from sklearn.cluster import KMeans

test_dataset_clean = torchvision.datasets.ImageFolder('/Users/ima029/Desktop/Unsupervised foraminifera groupings/Data/CROPS_Gol-F-30-3, 19-20_zoom 35/imagefolder')

# 1, 8, 10

c = 8

x_lab = x
y_lab = y
x_un = x_test

x_tot = np.concatenate([x_lab, x_un], axis=0)


centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 100)

kmeans = KMeans(n_clusters=100, random_state=0, init=centroids)
kmeans.fit(x_tot)

z = kmeans.labels_[:len(x_lab)]
print(np.unique(z, return_counts=True))

c_pred = kmeans.labels_[len(x_lab):]


z[y == 10]
c_pred[c_pred == 65]

output_dir = "cluster_all"


a = np.unique(c_pred, return_counts=True)

b = np.unique(z, return_counts=True)

# find the intersection between the two
intersection = np.intersect1d(a[0], b[0])


output_dir = "cluster_all"

clean_images = dataset_clean

n1 = 0
n = 0

for i in intersection:
    idx = np.where(c_pred == i)[0]
    images = [test_dataset_clean[i] for i in idx]
    p = y_pred[idx]
    os.makedirs(f"{output_dir}/{i}", exist_ok=True)
    for j, (img, _) in enumerate(images):
        name = dataset.classes[int(p[j])]
        img.save(f"{output_dir}/{i}/{name}.png")
    os.makedirs(f"{output_dir}/{i}_lab", exist_ok=True)
    images = [clean_images[i] for i in np.where(z == i)[0]]
    labels = y[np.where(z == i)[0]]
    n += len(images)
    for j, (img, _) in enumerate(images):
        name = dataset.classes[int(labels[j])]
        img.save(f"{output_dir}/{i}_lab/{name}.png")




