import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))

import json
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
import vit.vit_utils as vit_utils
import vit.vision_transformer as vits
from utils import LinearClassifier
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="mps")
parser.add_argument("--backbone_weights", type=str, default='/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint.pth')
parser.add_argument("--classifier_weights", type=str, default="")
parser.add_argument("--data_dir", type=str, default="data/labelled imagefolders/imagefolder_20")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--backbone_arch", type=str, default="vit_small")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation_split", type=float, default=None)
args = parser.parse_args()


# load the data
print("Loading data...")
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)

_, idx_val = train_test_split(np.arange(len(dataset)), test_size=args.validation_split, stratify=dataset.targets, random_state=args.seed)

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, idx_val),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

pretrained_model = vits.__dict__[args.backbone_arch](patch_size=16, num_classes=0, img_size=[224])
vit_utils.load_pretrained_weights(pretrained_model, args.backbone_weights, "teacher", args.backbone_arch, 16)

input_dim = {"vit_small": 384, "vit_base": 768}[args.backbone_arch]

classifier = LinearClassifier(input_dim, 20)
classifier.load_state_dict(torch.load(args.classifier_weights, map_location=args.device))

model = torch.nn.Sequential(pretrained_model, classifier).to(args.device)

model.eval()

os.makedirs(args.output_dir, exist_ok=True)

y_true = []
y_pred = []
y_loss = []
y_entr = []
with torch.no_grad():
    for x, y in dataloader:
        x = x.to(args.device)
        y = y.to(args.device)
        logits = model(x)
        proba = torch.nn.functional.softmax(logits, dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_loss.append(log_loss(y.cpu().numpy(), proba.cpu().numpy(), labels=np.arange(20)))
        y_entr.append(-torch.sum(proba * torch.log(proba), dim=1).cpu().numpy())
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
y_loss = np.mean(y_loss)
y_entr = np.concatenate(y_entr)

report = classification_report(
    y_true, y_pred, target_names=dataloader.dataset.dataset.classes, output_dict=True,
    )

pd.DataFrame(report).to_csv(os.path.join(args.output_dir, f"classification_report.csv"))

e = {}
for i in range(20):
    x = y_entr[y_true == i]
    e[dataloader.dataset.dataset.classes[i]] = x.tolist()

json.dump(e, open(os.path.join(args.output_dir, f"entropy.json"), "w"))

with open(os.path.join(args.output_dir, f"loss.txt"), "w") as f:
    f.write(f"Loss: {y_loss}\n")