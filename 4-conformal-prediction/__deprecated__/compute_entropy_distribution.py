import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))

import torch
import vit.vit_utils as vit_utils
import torchvision
import numpy as np
import pandas as pd
import vit.vision_transformer as vits
from tqdm import tqdm
from PIL import Image
from utils import LinearClassifier
from torchvision import transforms

backbone_weights = "/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint.pth"
classifier_weights = "./postprocessing/trained_models/vit_small/classifier_20241216122634.pth"
data_dir = "./data/labelled imagefolders/imagefolder_20"
batch_size = 32
device = "mps"

#transform = transforms.Compose([
#    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
#    transforms.RandomHorizontalFlip(p=0.5),
#    vit_utils.GaussianBlur(p=0.5),
#    transforms.ToTensor(),
#    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#])

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    vit_utils.GaussianBlur(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

backbone = vits.__dict__["vit_small"](patch_size=16, num_classes=0, img_size=[224])
vit_utils.load_pretrained_weights(backbone, backbone_weights, "teacher", "vit_small", 16)

classifier = LinearClassifier(384, 20)
classifier.load_state_dict(torch.load(classifier_weights, map_location=device))

model = torch.nn.Sequential(backbone, classifier).to(device)
model.eval()

e = []
y = []
p = []
for i in tqdm(range(10)):
    for images, labels in dataloader:
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            e.extend(entropy.cpu().numpy())
            y.extend(labels.numpy())
            p.extend(probs.argmax(dim=-1).cpu().numpy())
e = np.array(e)
y = np.array(y)
p = np.array(p)

d = {}
for i in range(20):
    subset = e[y == i]
    subset = np.random.choice(subset, 100, replace=False)
    d[i] = np.sort(subset)

pd.DataFrame(d).to_csv("entropy_distribution.csv", index=False)
