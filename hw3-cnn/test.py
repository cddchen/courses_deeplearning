import re
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class CustomImageDataSet(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = []
        self.img_dir = img_dir
        self.img_dirs = []
        self.transform = transform
        self.target_transform = target_transform

        pattern = re.compile(r'\d+')
        files = os.listdir(img_dir)
        for filename in files:
            # print(filename)
            self.img_dirs.append(filename)
            searchObj = pattern.findall(filename)
            # print(searchObj)
            self.img_labels.append(searchObj[0])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dirs[idx])
        image = read_image(img_path)
        label = int(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 25, 4)
        # initial size(224x224x3) -> conv1(220x220x6) -> pool(110x110x6) -> conv2(106x106x16) -> pool(53x53x16) -> conv3(50x50x25) -> pool(25x25x25) -> flatten(15625)
        self.fc1 = nn.Linear(15625, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    test_data = CustomImageDataSet('./food-11/validation', transform=transform)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
    PATH = './model.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    #images, labels = next(iter(test_loader))
    #outputs = net(images)
    #out = torchvision.utils.make_grid(images)
    #imshow(out, title=labels)
    #_, predicted = torch.max(outputs, 1)
    #print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))

    correct_pred = 0
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct_pred += (label == predicted).sum().item()

    print('Accuracy: {:.1f}%'.format(100 *  correct_pred / len(test_data)))