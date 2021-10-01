import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    '''
    Loss = \alphaT^2\times KL(\frac{Teacher's Logits}{T}||\frac{Student's Logits}{T})+(1-\alpha)(Original Loss)
    :param outputs: model's outputs
    :param labels: data's labels
    :param teacher_outputs: teacher model's outputs
    :param T: super weight
    :param alpha: importance of teacher model teaching
    :return: loss function
    '''
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                    F.softmax(teacher_outputs / T, dim=1)) \
                * (alpha * T * T)
    return hard_loss + soft_loss


# Knowledge Distillation
def run_epoch_knowledge_distillation(teacher_net, student_net, dataloader, optimizer, update=True, alpha=0.5):
    '''
    student_net study from teacher_net and normal cross entropy
    :param teacher_net:
    :param student_net:
    :param dataloader:
    :param optimizer:
    :param update: train or valid
    :param alpha: alpha of loss function definition
    :return: loss and hit
    '''
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.to(device)
        hard_labels = torch.LongTensor(hard_labels).to(device)
        # becuz Teacher_net dont need do backprop, use torch.no_grad
        # torch dont save the temporary variance
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            # do validation, use no_grad to save memory
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# Network Pruning
def run_epoch_network_pruning(net, dataloader, criterion, optimizer, update=True):
    '''
    the net study weight after pruning
    :param net:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param update: train or valid
    :return: loss and hit
    '''
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num
