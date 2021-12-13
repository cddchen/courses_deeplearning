import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from data import Omniglot
from model import Classifier, create_label, MAML, FirstOrderMAML
from utils import get_meta_batch

if __name__ == '__main__':
    n_way = 5
    k_shot = 1
    q_query = 1
    inner_train_steps = 1
    inner_lr = 0.4
    meta_lr = 0.001
    meta_batch_size = 32
    max_epoch = 40
    eval_batches = test_batches = 20
    train_data_path = './Omniglot/images_background/'
    test_data_path = './Omniglot/images_evaluation/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Omniglot(train_data_path, k_shot, q_query)
    train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200, 656])
    train_loader = DataLoader(train_set, batch_size=n_way, num_workers=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=n_way, num_workers=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query), batch_size=n_way, num_workers=8, shuffle=True, drop_last=True)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    meta_model = Classifier(1, n_way).to(device)
    # meta_model = torch.nn.parallel.DataParallel(meta_model)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(max_epoch):
        print("Epoch %d" % (epoch))
        train_meta_loss = []
        train_acc = []
        for step in tqdm(range(len(train_loader) // (meta_batch_size))):
            x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
            meta_loss, acc = FirstOrderMAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)
        print("Loss:", np.mean(train_meta_loss), "Accuracy:", np.mean(train_acc))

        val_acc = []
        for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
            x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
            _, acc = FirstOrderMAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps=3, train=False)
            val_acc.append(acc)
        print("Validation accuracy:", np.mean(val_acc))

        # test_acc = []
        # for test_step in tqdm(range(len(test_loader) // (test_batches))):
        #     x, val_iter = get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
        #     _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps=3, train=False)
        #     test_acc.append(acc)
        # print("Test Accuracy:", np.mean(test_acc))