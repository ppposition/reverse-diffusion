import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
import numpy as np
test_data = torchvision.datasets.CIFAR10(root="../input/cifar10-python",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

images = test_data[1][0].squeeze(0).permute(1, 2, 0).numpy()
images = (images * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(images, 'RGB').save('testdata/test5.png')
