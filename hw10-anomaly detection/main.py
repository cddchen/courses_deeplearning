import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import save_model, set_device
from data import preprocess, ImgDataSet
from model import AE, AutoEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans


def KNN(data, test_data):
    data = data.reshape(len(data), -1)
    test_data = test_data.reshape(len(test_data), -1)
    scores = list()
    for n in range(1, 10):
        kmeans_data = MiniBatchKMeans(n_clusters=n,
                                      batch_size=100).fit(data)
        test_cluster = kmeans_data.predict(test_data)
        test_dist = np.sum(np.square(kmeans_data.cluster_centers_[test_cluster] - test_data), axis=1)
        test_pred = test_dist

    #     score = f1_score(test_label, test_pred, average='micro')
    #     score = roc_auc_score(test_label, test_pred, average='micro')
    #     scores.append(score)
    # print(np.max(scores), np.argmax(scores))
    # print(scores)
    # print('auc score: {}'.format(np.max(scores)))


def pca(data, test_data):
    data = data.reshape(len(data), -1)
    test_data = test_data.reshape(len(test_data), -1)
    pca = PCA(n_components=2).fit(data)

    test_data_projected = pca.transform(test_data)
    test_data_reconstructed = pca.inverse_transform(test_data_projected)
    dist = np.sqrt(np.sum(np.square(test_data_reconstructed - test_data).reshape(len(test_data), -1), axis=1))

    test_data_pred = dist
    score = roc_auc_score(test_data_label, test_data_pred, average='micro')
    score = f1_score(test_data_label, test_data_pred, average='micro')
    print('auc score: {}'.format(score))


if __name__ == '__main__':
    print('Loading data...')
    train_data = np.load('./train.npy')
    test_data = np.load('./test.npy')
    train_data, test_data = preprocess(train_data), preprocess(test_data)
    dataset = ImgDataSet(train_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    print('Loading model...')
    model, args = set_device(AutoEncoder())
    model.load_state_dict(torch.load('AE_last.pth'))
    args['epochs'] = 100
    args['samples'] = 5
    print("Start eval...")

    # KNN
