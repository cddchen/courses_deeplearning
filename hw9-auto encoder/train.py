import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from units import same_seeds
from data import preprocess, ImgDataSet
from model import AE


def train_loop(autoencoder, dataloader, args):
    epochs = args['epochs']
    device = args['device']
    n_gpu = args['n_gpu']
    gpu_ids = args['gpu_ids']
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
            loss = criterion(out, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if n_gpu > 1:
                loss = loss.mean()
            mse_loss += loss.detach().item()

        if (epoch + 1) % 10 == 0:
            losses.append(mse_loss)
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
            input = (input[0].detach().cpu().numpy() + 1.) / 2.
            input = np.transpose(input, (1, 2, 0))
            inputs.append(input)
            output = (output[0].detach().cpu().numpy() + 1.) / 2.
            output = np.transpose(output, (1, 2, 0))
            outputs.append(output)

        return inputs, outputs


if __name__ == '__main__':
    same_seeds(0)
    data = preprocess(np.load('./trainX_new.npy'))
    dataset = ImgDataSet(data)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=True)

    print("training.")
    print("-" * 50)
    epochs = 100
    model = AE()

    print('Setting cuda&cpu...')
    device = torch.device('cpu')
    n_gpu = 0
    gpu_ids = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_ids = list(range(0, n_gpu))
        if n_gpu > 1:
            model = torch.nn.DataParallel(
                model, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            print('-> GPU training available! Training will use GPU(s) {}\n'.format(gpu_ids))
        device = torch.device('cuda')
    print('device: ', device)

    model = model.to(device)
    args = {'epochs': 100, 'device': device, 'n_gpu': n_gpu, 'gpu_ids': gpu_ids, 'samples': 5}

    losses = train_loop(model, dataloader, args)

    print("-" * 50)
    print('saving model to auto-encoder.pth')
    torch.save(model.state_dict(), f'./auto-encoder_new-struct.pth')
    print('done.')

    plt.plot(losses)
    plt.show()
