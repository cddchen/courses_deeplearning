import math
import re
import time

import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from model import Classifier
from units import readfile
from data import CustomImageDataSet


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = '/content/drive/MyDrive/model(2).pth'
    model = torch.load(path)
    model = model.to(device)
    model.eval()

    workspace_dir = './food-11'
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         # Tensor format (channel, width, height)
         transforms.ToTensor()]
    )

    batch_size = 1
    test_set = CustomImageDataSet(train_x, train_y, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    images, label = next(iter(test_loader))
    print('images size:', images.size())
    # set the requires_grad_ to the image for retrieving gradients
    images.requires_grad_()

    output = model(images.to(device))
    output_max_idx = output.argmax()
    output_max = output[0, output_max_idx]
    print('output_label:', output_max_idx)
    print('true label:', label)

    output_max.backward()

    # use the abs of grad
    saliency, _ = torch.max(images.grad.data.abs(), dim=1)
    saliency = saliency.reshape(128, 128)

    # convert (channel, width, height) -> (width, height, channel)
    images = images.reshape(-1, 128, 128)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()


