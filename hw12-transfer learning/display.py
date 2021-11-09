import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import resnet45, LabelClassifier, DomainDiscriminator
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--log_save_dir", type=str, default='logs', help="log files save")
    parser.add_argument("--checkpoint_save_dir", type=str, default='checkpoints', help="log files save")
    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.log_save_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_save_dir, exist_ok=True)

    source_transform = transforms.Compose([
        # 轉灰階: Canny 不吃 RGB。
        transforms.Grayscale(),
        # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # 重新將np.array 轉回 skimage.Image
        transforms.ToPILImage(),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
        transforms.RandomRotation(15, fill=(0,)),
        # 最後轉成Tensor供model使用。
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        # 轉灰階: 將輸入3維壓成1維。
        transforms.Grayscale(),
        # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
        transforms.Resize((32, 32)),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
        transforms.RandomRotation(15, fill=(0,)),
        # 最後轉成Tensor供model使用。
        transforms.ToTensor(),
    ])
    source_dataset = ImageFolder('train_data', transform=source_transform)
    target_dataset = ImageFolder('test_data', transform=target_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=opt.batch_size)
    target_dataloader = DataLoader(target_dataset, batch_size=opt.batch_size)

    resnet = resnet45()
    resnet.train()
    discriminator = DomainDiscriminator()
    discriminator.train()
    classifier = LabelClassifier()
    classifier.train()

    print('Setting cuda&cpu...')
    device = torch.device('cpu')
    n_gpu = 0
    gpu_ids = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_ids = list(range(0, n_gpu))
        device = torch.device('cuda')
        if n_gpu > 1:
            resnet = torch.nn.DataParallel(
                resnet, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            resnet = resnet.to(device)
            discriminator = torch.nn.DataParallel(
                discriminator, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            discriminator = discriminator.to(device)
            classifier = torch.nn.DataParallel(
                classifier, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            classifier = classifier.to(device)
            print('-> GPU training available! Training will use GPU(s) {}'.format(gpu_ids))

    resnet.load_state_dict(torch.load(f'checkpoints/resnet_200.pth'))
    discriminator.load_state_dict(torch.load(f'checkpoints/discriminator_200.pth'))
    classifier.load_state_dict(torch.load(f'checkpoints/classifier_200.pth'))

    source_feature = None
    for data in source_dataloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        sz = imgs.size(0)

        features = resnet(imgs)
        features = features.reshape(sz, -1)

        if source_feature is None:
            source_feature = features.detach().cpu().numpy()
        else:
            source_feature = np.concatenate((source_feature, features.detach().cpu().numpy()), axis=0)
    print('source feature shape:', source_feature.shape)

    target_feature = None
    for i, data in enumerate(target_dataloader):
        if i * 256 > 5000:
            break
        imgs, _ = data
        imgs = imgs.to(device)
        sz = imgs.size(0)

        features = resnet(imgs)
        features = features.reshape(sz, -1)

        if target_feature is None:
            target_feature = features.detach().cpu().numpy()
        else:
            target_feature = np.concatenate((target_feature, features.detach().cpu().numpy()), axis=0)
    print('target feature shape:', target_feature.shape)

    pca = PCA(n_components=2).fit(source_feature)
    source_project = pca.transform(source_feature)
    target_project = pca.transform(target_feature)

    plt.scatter(source_project[:, 0], source_project[:, 1], c='r')
    plt.scatter(target_project[:, 0], target_project[:, 1], c='b')
    plt.show()