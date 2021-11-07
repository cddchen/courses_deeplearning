import argparse
import os.path
import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from data import get_dataset, FaceDataset
from utils import set_device, same_seeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--log_save_dir", type=str, default='logs', help="log files save")
    parser.add_argument("--checkpoint_save_dir", type=str, default='checkpoints', help="log files save")
    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.log_save_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_save_dir, exist_ok=True)

    print('-' * 30)
    print('Loading model...')
    G = Generator(in_dim=opt.latent_dim)
    D = Discriminator(3)
    print('Setting cuda&cpu...')
    device = torch.device('cpu')
    n_gpu = 0
    gpu_ids = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_ids = list(range(0, n_gpu))
        device = torch.device('cuda')
        if n_gpu > 1:
            G = torch.nn.DataParallel(
                G, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            G = G.to(device)
            D = torch.nn.DataParallel(
                D, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            D = D.to(device)
            print('-> GPU training available! Training will use GPU(s) {}'.format(gpu_ids))
    # G.load_state_dict(torch.load('checkpoints/nsgan_g_30.pth'))
    # D.load_state_dict(torch.load('checkpoints/nsgan_d_30.pth'))
    # print('Start to epochs 30:')
    G.train()
    D.train()

    criterion = nn.BCELoss()

    opt_D = optim.Adam(D.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

    same_seeds(0)

    dataset = get_dataset('./faces')
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    z_sample = Variable(torch.randn(100, opt.latent_dim)).to(device)

    print('-' * 30)
    print('Start to train model.')
    for e, epoch in enumerate(range(opt.n_epochs)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # Training D
            z = Variable(torch.randn(bs, opt.latent_dim)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = G(z).to(device)

            # label
            r_label = torch.ones((bs)).to(gpu_ids[-1])
            f_label = torch.zeros((bs)).to(gpu_ids[-1])

            # dis
            r_logit = D(r_imgs.detach()).to(gpu_ids[-1])
            f_logit = D(f_imgs.detach()).to(gpu_ids[-1])

            # compute loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(1.0 - f_logit, r_label)
            loss_D = r_loss + f_loss

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Training G
            # leaf
            z = Variable(torch.randn(bs, opt.latent_dim)).to(device)
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = criterion(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(f'\rEpoch [{epoch+1}/{opt.n_epochs}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(opt.log_save_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f'| Save some samples to {filename}.')
        # show generated img
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e+1) % 5 == 0:
            torch.save(G.state_dict(), f'checkpoints/nsgan_g_{e+1}.pth')
            torch.save(D.state_dict(), f'checkpoints/nsgan_d_{e+1}.pth')
            print('Save model successful.')



