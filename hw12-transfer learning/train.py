import os
import argparse
import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import resnet45, LabelClassifier, DomainDiscriminator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--log_save_dir", type=str, default='logs', help="log files save")
    parser.add_argument("--checkpoint_save_dir", type=str, default='checkpoints', help="log files save")
    parser.add_argument("--lamb", type=float, default='0.1', help="control the adversarial loss")
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

    domain_criterion = nn.BCEWithLogitsLoss()
    label_criterion = nn.CrossEntropyLoss()
    opt_resnet = optim.Adam(resnet.parameters(), lr=opt.learning_rate)
    opt_D = optim.Adam(discriminator.parameters(), lr=opt.learning_rate)
    opt_C = optim.Adam(classifier.parameters(), lr=opt.learning_rate)

    print("=" * 30)
    print('Start to train...')
    for e, epoch in enumerate(range(opt.n_epochs)):
        for i, ((source_imgs, source_labels), (target_imgs, _)) in enumerate(zip(source_dataloader, target_dataloader)):
            source_imgs, source_labels, target_imgs = source_imgs.to(gpu_ids[-1]), source_labels.to(gpu_ids[-1]), target_imgs.to(gpu_ids[-1])
            mixed_imgs = torch.cat([source_imgs, target_imgs], dim=0)
            domain_label = torch.zeros([source_imgs.shape[0] + target_imgs.shape[0]]).to(gpu_ids[-1])
            domain_label[:source_imgs.shape[0]] = 1
            # train the discriminator
            feature = resnet(mixed_imgs)
            domain_logits = discriminator(feature.detach())
            loss = domain_criterion(domain_logits, domain_label)
            domain_loss_item = loss.item()
            loss.backward()
            opt_D.step()
            # train the resnet
            class_logits = classifier(feature[:source_imgs.shape[0]])
            domain_logits = discriminator(feature)
            loss = label_criterion(class_logits, source_labels) - opt.lamb * domain_criterion(domain_logits, domain_label)
            resnet_loss_item = loss.item()
            loss.backward()
            opt_resnet.step()
            opt_C.step()

            opt_C.zero_grad()
            opt_D.zero_grad()
            opt_resnet.zero_grad()

            total_hit = torch.sum(torch.argmax(class_logits, dim=1) == source_labels).item()
            total_num = source_imgs.shape[0]

            print(f'\rEpoch [{epoch+1}/{opt.n_epochs}] {i+1}/{len(source_dataloader)} Loss_D: {domain_loss_item:.4f} Loss_resnet: {resnet_loss_item:.4f} acc: {total_hit/total_num:6.4f}', end='')
        if (e+1) % 50 == 0:
            torch.save(resnet.state_dict(), f'checkpoints/resnet_{e+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_{e+1}.pth')
            torch.save(classifier.state_dict(), f'checkpoints/classifier_{e + 1}.pth')
