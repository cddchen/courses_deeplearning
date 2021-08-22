import re
import time
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


class CustomImageDataSet(Dataset):
    def __init__(self, x, y=None, transform=None, target_transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[idx]
            if self.target_transform is not None:
                Y = self.target_transform(Y)
            return X, Y
        else:
            return X


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input size(3, 128, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),    # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),   # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),   # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),   # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


if __name__ == '__main__':
    workspace_dir = './food-11'
    print("Reading data...")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))

    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15),
         transforms.ToTensor()]
    )
    test_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor()]
    )
    batch_size = 64
    training_data = CustomImageDataSet(train_x, train_y, train_transform)
    val_data = CustomImageDataSet(val_x, val_y, test_transform)
    test_data = CustomImageDataSet(test_x, transform=test_transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = Classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 30
    # training
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.

        model.train()
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % (
            epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / training_data.__len__(),
            train_loss / training_data.__len__(), val_acc / val_data.__len__(), val_loss / val_data.__len__()
        ))

    # 利用train和eval数据联合训练增强网络强度
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    train_val_set = CustomImageDataSet(train_val_x, train_val_y, transform=train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

    model_best = Classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
    num_epoch = 30

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.

        model.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' %
              (epoch+1, num_epoch, time.time()-epoch_start_time, train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            test_pred = model_best(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    with open('predict.csv', 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))


    
