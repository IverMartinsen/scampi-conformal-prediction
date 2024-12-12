import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "vit"))

import torch
import argparse
import vit.vit_utils as vit_utils
import torchvision
import numpy as np
import vit.vision_transformer as vits
from time import time
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--pretrained_weights", type=str, default="")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--data_dir", type=str, default="data/labelled imagefolders/imagefolder_20")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()


# load the data
print("Loading data...")
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    vit_utils.GaussianBlur(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)

i_tr, i_val = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.targets)

x_tr = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, i_tr),
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers,
    pin_memory=True,
)
x_val = torch.utils.data.DataLoader(
    torch.utils.data.Subset(dataset, i_val),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

print(f"Number of training images: {len(i_tr)}")
print(f"Number of validation images: {len(i_val)}")

# load the model
pretrained_model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, img_size=[224])
vit_utils.load_pretrained_weights(pretrained_model, args.pretrained_weights, "teacher", "vit_small", 16)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


classifier = LinearClassifier(384, 20)

model = torch.nn.Sequential(pretrained_model, classifier).to(args.device)

for param in pretrained_model.parameters():
    param.requires_grad = False

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# train the model
def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0
    t = time()
    for x, y in dataloader:
        x = x.to(args.device)
        y = y.to(args.device)

        # optimizer.zero_grad(): Clear the gradients of all optimized torch.Tensor.
        # This involves setting the .grad attribute of all parameters to None.
        optimizer.zero_grad()

        # Tensor.backward(): Compute the gradients of Tensor w.r.t. model parameters.
        # The gradients are stored in the .grad attribute of the parameters.
        logits = model(x)
        loss = loss_fn(logits, y)
        correct += (logits.argmax(1) == y).type(torch.float).sum().item()
        loss.backward()

        # optimizer.step(): Perform a single optimization step (parameter update).
        # Updates each parameter wrt the gradient stored in the parameter's .grad attribute.
        optimizer.step()
    
    accuracy = correct / size
    
    return loss.item(), accuracy, time() - t

def test_model(model, dataloader, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable gradient calculation to speed up computation
        for x, y in dataloader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    return test_loss, correct

def train_model(model, x_tr, loss_fn, optimizer, num_epochs, x_val=None):
    for epoch in range(num_epochs):
        loss, acc, t = train_one_epoch(model, x_tr, loss_fn, optimizer)
        loss, acc, t = round(loss, 4), round(acc, 4), round(t, 4)
        if x_val is not None:
            test_loss, test_acc = test_model(model, x_val, loss_fn)
            val_msg = f", Test Loss: {test_loss}, Test Accuracy: {test_acc}"
        else:
            val_msg = ""
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}, Accuracy: {acc}, Time: {t}" + val_msg)

if __name__ == "__main__":
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, x_tr, loss_fn, optimizer, args.epochs, x_val)

    torch.save(classifier.state_dict(), "classifier.pth")