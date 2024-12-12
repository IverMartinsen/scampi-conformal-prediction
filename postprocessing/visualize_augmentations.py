import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

data_dir = "data/labelled imagefolders/imagefolder_20"

images = []
values = []

for value in np.arange(0, 1.1, 0.1):
    print(f"Processing value: {value:.1f}...")
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=value, contrast=0.0, saturation=0.0, hue=0),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )

    for i in range(10):
        img = dataset[i][0].permute(1, 2, 0).numpy()
        images.append(img)
        values.append(value)

fig, axs = plt.subplots(len(values) // 10, 10, figsize=(20, 20))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i])
    ax.axis("off")
    ax.set_title(f"Brightness: {values[i]:.1f}")

plt.tight_layout()
plt.savefig("brightness_variation.png", dpi=300)
plt.close()