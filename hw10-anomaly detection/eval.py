import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import save_model, set_device
from data import preprocess, ImgDataSet
from model import VAE, AutoEncoder


def reconstruct_img(autoencoder, dataset, args):
    device = args['device']
    n_gpu = args['n_gpu']
    gpu_ids = args['gpu_ids']
    samples = args['samples']
    autoencoder.eval()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True)
    inputs, outputs = [], []
    with torch.no_grad():
        for sample in range(samples):
            batch = next(iter(dataloader))
            if n_gpu > 1:
                batch = batch.to(gpu_ids[-1])
            else:
                batch = batch.to(device)
            hidden, output = autoencoder(batch)
            inputs.append(batch[0].detach().cpu().numpy())
            outputs.append(output[0].detach().cpu().numpy())

    return inputs, outputs


def eval(autoencoder, data, args):
    device = args['device']
    n_gpu = args['n_gpu']
    gpu_ids = args['gpu_ids']
    autoencoder.eval()
    dataset = ImgDataSet(data)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)
    reconstruct = list()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if n_gpu > 1:
                batch = batch.to(gpu_ids[-1])
            else:
                batch = batch.to(device)
            hidden, output = autoencoder(batch)
            reconstruct.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstruct, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - data).reshape(len(dataset), -1), axis=1))
    y_pred = anomality
    with open('prediction.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(dataset)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))


if __name__  == '__main__':
    print('Loading data...')
    train_data = np.load('./train.npy')
    test_data = np.load('./test.npy')
    train_data, test_data = preprocess(train_data), preprocess(test_data)
    dataset = ImgDataSet(train_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    print('Loading model...')
    model, args = set_device(AutoEncoder())
    model.load_state_dict(torch.load('AE_last_11.1.pth'))
    args['epochs'] = 100
    args['samples'] = 5
    print("Start eval...")
    # # save_model(model, 'AE', args)
    # max_len = eval(model, dataset, None, args)
    # print('max distance in group:', max_len)
    # dises = eval(model, test_dataset, max_len, args)
    # labels = [True if dis <= max_len else False for dis in dises]
    # print(labels)

    inputs, outputs = reconstruct_img(model, dataset, args)
    plt.figure(figsize=(32, 32))
    samples = args['samples']
    for sample in range(samples):
        input = inputs[sample]
        output = outputs[sample]
        input = np.transpose(input, (1, 2, 0))
        output = np.transpose(output, (1, 2, 0))
        plt.subplot(samples, 2, sample * 2 + 1)
        plt.imshow(input)
        plt.subplot(samples, 2, sample * 2 + 2)
        plt.imshow(output)
    plt.tight_layout()
    plt.show()
    eval(model, test_data, args)
