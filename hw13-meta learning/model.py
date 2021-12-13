from collections import OrderedDict
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        x = self.logits(x)
        return x
    
    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的参数，也就是convolution的weight和bias，以及batchnormalization的weight和bias，这是一个OrderedDict
        '''
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'], params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()

def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps=1, inner_lr=0.4, train=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size, n_way*(k_shot+q_query), 1, 28, 28]
    n_way: 每个分类的task要有几个class
    k_shot: 每个类别在training的时候会有多少张照片
    q_query: 在testing时，每个类别会有多少张照片
    """
    criterion = loss_fn
    task_loss = []
    task_acc = []
    for meta_batch in x:
        train_set = meta_batch[:n_way*k_shot] # 用来更新inner loop的参数
        val_set = meta_batch[n_way*k_shot:] # 用来更新outer loop的参数

        fast_weights = OrderedDict(model.named_parameters())
        
        for inner_step in range(inner_train_steps):
            train_label = create_label(n_way, k_shot).to(device)
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict((name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        
        val_label = create_label(n_way, q_query).to(device)
        logits = model.functional_forward(val_set, fast_weights)
        loss = criterion(logits, val_label)
        task_loss.append(loss)
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()
        task_acc.append(acc)
    
    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


def FirstOrderMAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_steps=1, inner_lr=0.4, train=True,
         device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size, n_way*(k_shot+q_query), 1, 28, 28]
    n_way: 每个分类的task要有几个class
    k_shot: 每个类别在training的时候会有多少张照片
    q_query: 在testing时，每个类别会有多少张照片
    """
    criterion = loss_fn
    task_loss = []
    task_acc = []
    for meta_batch in x:
        train_set = meta_batch[:n_way * k_shot]  # 用来更新inner loop的参数
        val_set = meta_batch[n_way * k_shot:]  # 用来更新outer loop的参数

        fast_weights = OrderedDict(model.named_parameters())

        for inner_step in range(inner_train_steps):
            train_label = create_label(n_way, k_shot).to(device)
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=False, retain_graph=False)
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

        val_label = create_label(n_way, q_query).to(device)
        logits = model.functional_forward(val_set, fast_weights)
        loss = criterion(logits, val_label)
        task_loss.append(loss)
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc