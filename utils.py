import torch
import torchvision


def read_fn(bytes):
    image = torch.tensor(bytes) # sequence of bytes
    image = torchvision.io.decode_png(image) # shape: (3, H, W)
    image = image.permute(1, 2, 0) # shape: (H, W, 3)
    return image.numpy()
