import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


def network_slimming(old_model, new_model):
    '''
    按照new_model的参数需要从old_model中复制更重要的参数
    :param old_model: 老模型
    :param new_model: 新模型
    :return: 新模型
    '''
    params = old_model.state_dict()
    new_params = new_model.state_dict()

    # selected_idx <- selected neuron index in each layer
    selected_idx = []
    for i in range(8):
        # BatchNorm's gamma in cnn.{i}.1.weight
        importance = params[f'cnn.{i}.1.weight']
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        ranking = torch.argsort(importance, descending=True)
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # if the layer is cnn, 移植参数
        # if the layer is fc or just a number, 直接复制
        if name.startwith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # when process to pointwise layer, make now_processed+1, tag this layer have processed
            if name.startwith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # if this layer is pointwise, weight will be influenced by results of before and after
            # notice the format of Conv2d(x,y,1) in weight shape is (y,x,1,1), so you need to switch
            if name.endswith('3.weight'):
                # the output neuron of last layer don't need prune
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:, selected_idx[now_processed - 1]]
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    new_model.load_state_dict(new_params)
    return new_model
