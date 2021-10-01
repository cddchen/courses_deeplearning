import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from data import get_dataloader
from models import StudentNet
from train import run_epoch_knowledge_distillation

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    train_dataloader = get_dataloader('training', batch_size=batch_size)
    valid_dataloader = get_dataloader('validation', batch_size=batch_size)

    # define the models
    teacher_net = models.resnet18(pretrained=False, num_classes=11).to(device)
    student_net = StudentNet(base=16).to(device)
    # load the pretrained model weights
    teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
    optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)

    teacher_net.eval()
    now_best_acc = 0
    for epoch in range(200):
        student_net.train()
        train_loss, train_acc = run_epoch_knowledge_distillation(
            teacher_net, student_net, train_dataloader, optimizer, update=True
        )
        student_net.eval()
        valid_loss, valid_acc = run_epoch_knowledge_distillation(
            teacher_net, student_net, valid_dataloader, optimizer, update=False
        )

        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), 'student_model.bin')
        print('epoch %3d: train loss: %6.4f, acc: %6.4f; valid loss: %6.4f, acc: %6.4f'
              % (epoch, train_loss, train_acc, valid_loss, valid_acc))
