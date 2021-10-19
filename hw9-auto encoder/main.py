import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from model import AutoEncoder, AE
from data import ImgDataSet, preprocess
from train import train_loop, eval
from eval import inference, predict
from units import plot_scatter, cal_acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AE().to(device)
    model.load_state_dict(torch.load('auto-encoder-cpu.pth'))
    model.eval()

    # select some samples to show model ability
    samples = 5
    data = preprocess(np.load('./trainX_new.npy'))
    dataset = ImgDataSet(data)
    inputs, outputs = eval(model, device, samples, dataset)
    plt.figure(figsize=(32, 32))
    cnt = 0
    for sample in range(samples):
        cnt += 1
        plt.subplot(samples, 2, cnt)
        plt.imshow(inputs[sample])
        cnt += 1
        plt.subplot(samples, 2, cnt)
        plt.imshow(outputs[sample])
    plt.tight_layout()
    plt.show()

    # show the embedded result
    valX = np.load('valX.npy')
    valY = np.load('valY.npy')
    latents = inference(valX, model, device)
    pred_from_latent, emb_from_predict = predict(latents)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_predict, valY, savefig='p1_baseline.png')
