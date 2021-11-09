import matplotlib.pyplot as plt
import torch
import random
import numpy as np


def set_device(models) -> map:
    print('Setting cuda & cpu...')
    device = torch.device('cpu')
    n_gpu = 0
    gpu_ids = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_ids = list(range(0, n_gpu))
        device = torch.device('cuda')
        if n_gpu > 1:
            if type(models) is list:
                for model in models:
                    model = torch.nn.DataParallel(
                        model, device_ids=gpu_ids, output_device=gpu_ids[-1]
                    )
                    model = model.to(device)
            else:
                models = torch.nn.DataParallel(
                    models, device_ids=gpu_ids, output_device=gpu_ids[-1]
                )
                models = models.to(device)
            # print('-> GPU training available! Training will use GPU(s) {}'.format(gpu_ids))
    else:
        if type(models) is list:
            for model in models:
                model = model.to(device)
        else:
            models = models.to(device)
    args = {'device': device, 'n_gpu': n_gpu, 'gpu_ids': gpu_ids}
    print(args)
    return args


def save_model(model, dir, args):
    if args['n_gpu'] > 1:
        torch.save(model.state_dict(), dir+'.pth')
    model = model.to(torch.device('cpu'))
    torch.save(model, dir+'-cpu.pth')
    device = args['device']
    model = model.to(device)


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