import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import save_model, set_device
from data import preprocess, ImgDataSet
from model import VAE, AutoEncoder


def train_loop(autoencoder, dataloader, args: dict):
    device = args['device']
    n_gpu = args['n_gpu']
    gpu_ids = args['gpu_ids']
    epochs = args['epochs']
    criterion = nn.MSELoss()
    learning_rate = 1e-5
    decay = 1e-5
    opt = torch.optim.Adam(autoencoder.parameters(),
                           lr=learning_rate,
                           weight_decay=decay)

    autoencoder.train()
    losses = []
    for epoch in range(epochs):
        mse_loss = 0
        for batch in dataloader:
            if n_gpu > 1:
                batch = batch.to(gpu_ids[-1])
            else:
                batch = batch.to(device)
            hidden, out = autoencoder(batch)
            # if model is vae:
            # loss = loss_VAE(out[0], batch, out[1], out[2], criterion)
            loss = criterion(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if n_gpu > 1:
                loss = loss.mean()
            mse_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            losses.append(mse_loss)
        print("epoch [%d/%d], mse loss: %.4f" % (epoch, epochs, mse_loss))

    save_model(autoencoder, f'AE_last_11.1', args)
    return losses


if __name__ == '__main__':
    print('Loading data...')
    train_data = np.load('./train.npy', allow_pickle=True)
    test_data = np.load('./test.npy', allow_pickle=True)
    dataset = ImgDataSet(preprocess(train_data))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    print('Loading model...')
    model, args = set_device(AutoEncoder())
    args['epochs'] = 100
    args['samples'] = 5
    print("Start eval...")
    losses = train_loop(model, dataloader, args)