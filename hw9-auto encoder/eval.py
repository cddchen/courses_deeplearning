import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from data import preprocess, ImgDataSet
from torch.utils.data import Dataset, DataLoader
from model import AE
import numpy as np


def inference(X, model, device, batch_size=256):
    X = preprocess(X)
    dataset = ImgDataSet(X)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.to(device))
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()),
                                     axis=0)
    print('Latents Shape:', latents.shape)
    return latents


def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200,
                            kernel='rbf',
                            n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # Second Dimension Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2,
                           random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded


def invert(pred):
    return np.abs(1-pred)


def save_prediction(pred, out_csv='prediciton.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediciton to {out_csv}.')


if __name__ == '__main__':
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
    model.load_state_dict(torch.load('auto-encoder_new-struct.pth'))
    
    trainX = np.load('trainX_new.npy')

    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    save_prediction(pred)

    save_prediction(invert(pred), 'prediction_invert.csv')
