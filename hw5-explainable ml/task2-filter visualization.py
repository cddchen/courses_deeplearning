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


# set the activation function
def _register_forward_hooks(layer, filter_idx):

  def _record_activation(module, input_, output):
    global activation
    activation = torch.mean(output[:, filter_idx, :, :])

  return layer.register_forward_hook(_record_activation)


# hook the input to catch gradient
def hook_input(input_tensor):
  def hook_function(grad_in):
    global gradients
    gradients = grad_in
  return input_tensor.register_hook(hook_function)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = '/content/drive/MyDrive/model.pth'
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
    activation = None
    gradients = None
    hooks = []

    input = np.uint8(np.random.uniform(150, 180, (128, 128, 3)))
    plt.imshow(input)
    plt.show()
    input_ = transform(input)
    input_ = input_.reshape(1, 3, 128, 128)
    input_.requires_grad = True
    gradients = torch.zeros(input_.shape)
    while len(hooks) > 0:
        hooks.pop().remove()
    hooks.append(_register_forward_hooks(model.conv1, 16))
    hooks.append(hook_input(input_))
    output = []
    num_iter = 100
    lr = 1.
    for i in range(num_iter):
        model(input_.to('cuda'))
        activation.backward()
        # can replaced by grad,_ = autograd.grad(activation, input_)
        gradients /= (torch.sqrt(torch.mean(torch.mul(gradients, gradients))) + 1e-5)
        input_ = input_ + gradients * lr
        output.append(input_)
    plt.imshow(output[-1].detach().numpy().reshape(128, 128, 3))
    plt.show()