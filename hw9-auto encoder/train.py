import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from units import same_seeds
from data import preprocess, ImgDataSet
from model import AE


def train_loop(autoencoder, device, epochs, dataloader):
    criterion = nn.MSELoss()
    learning_rate = 1e-5
    decay = 1e-5
    opt = torch.optim.Adam(autoencoder.parameters(),
                           lr=learning_rate,
                           weight_decay=decay)

    autoencoder.train()
    mse_loss = 0
    losses = []
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            hidden, out = autoencoder(batch)
            loss = criterion(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            mse_loss += loss.detach().item()

        if (epoch + 1) % 10 == 0:
            losses.append(mse_loss)
            mse_loss = 0
        print("epoch %d, mse loss: %.2f" % (epoch, mse_loss))

    return losses


def eval(autoencoder, device, samples, dataset):
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True)
    inputs = []
    outputs = []
    with torch.no_grad():
        for sample in range(samples):
            input = next(iter(dataloader))
            input = input.to(device)
            hidden, output = autoencoder(input)
            input = input[0]
            input = np.transpose(input, (1, 2, 0))
            inputs.append(input)
            output = output[0]
            output = np.transpose(output, (1, 2, 0))
            outputs.append(output)

        return inputs, outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seeds(0)
    print('device:', device)
    data = preprocess(np.load('./trainX_new.npy'))
    dataset = ImgDataSet(data)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=True)

    print("training.")
    print("-" * 50)
    epochs = 500
    autoencoder = AE().to(device)

    losses = train_loop(autoencoder, device, epochs, dataloader)

    print("-" * 50)
    print('saving model to auto-encoder.pth')
    torch.save(autoencoder.state_dict(), f'./auto-encoder.pth')
    print('done.')

    plt.plot(losses)
    plt.show()
