import matplotlib.pyplot as plt
import torch
import random
import numpy as np


def save_model(model, dir):
    torch.save(model.state_dict(), dir+'.pth')
    model = model.to(torch.device('cpu'))
    model = model.to('cpu')
    torch.save(model, dir+'-cpu.pth')


def K_means(points: list, k: int) -> torch.Tensor:
    centers = torch.rand((k, len(points[0])))
    # = torch.zeros((points.shape[0], k))
    epochs = 10
    for epoch in range(epochs):
        # calculate the nearest center of each point
        dist = torch.zeros((len(points), k))
        for row in range(len(points)):
            for i in range(k):
                #print(points[row].shape)
                #print(centers[i].shape)
                dist[row, i] = sum((points[row] - centers[i]) ** 2).item()

        idxs = dist.argmax(dim=1)
        # re-calculate the centers
        for i in range(k):
            sum_points = torch.zeros((1, len(points[0])))
            cnt = 0
            for row in range(len(points)):
                if i == idxs[row]:
                    sum_points += points[i]
                    cnt += 1
            centers[i] = sum_points / cnt

    return centers


def count_parameters(model, only_trainable=False):
    '''
    count the number of parameters need to train
    :param model:
    :param only_trainable:
    :return:
    '''
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def same_seeds(seed):
    '''
    set the same seed to reproduce the result
    :param seed:
    :return:
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_acc(gt, pred):
    '''
    compute categorization accuracy of our task
    :param gt:  ground truth labels (9000, )
    :param pred: predicted labels (9000, )
    :return:
    '''
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return max(acc, 1-acc)


def plot_scatter(feat, label, savefig=None):
    '''
    plot scatter image
    :param feat: the (x, y) coordinate of clustering result, shape (9000, 2)
    :param label: ground truth label of image (0/1), shape (9000, 2)
    :param savefig:
    :return:
    '''
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c=label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return


if __name__ == '__main__':
    x = [[1, 1], [4, 2], [0, 2]]
    x = torch.tensor(x)
    centers = K_means(x, 2)
    print(centers)