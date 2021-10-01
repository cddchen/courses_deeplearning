import torch
import torch.nn as nn
import torch.optim as optim
from train import run_epoch_network_pruning
from slim import network_slimming
from data import get_dataloader
from models import StudentNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    train_dataloader = get_dataloader('training', batch_size=batch_size)
    valid_dataloader = get_dataloader('validation', batch_size=batch_size)

    net = StudentNet().to(device)
    # the net through knowledge distillation
    net.load_state_dict(torch.load('student_custom_small.bin'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    now_width_mult = 1
    for i in range(5):
        now_width_mult *= 0.95
        new_net = StudentNet(width_mult=now_width_mult).to(device)
        params = net.state_dict()
        net = network_slimming(net, new_net)
        now_best_acc = 0
        for epoch in range(5):
            net.train()
            train_loss, train_acc = run_epoch_network_pruning(net, train_dataloader, criterion, optimizer, update=True)
            net.eval()
            valid_loss, valid_acc = run_epoch_network_pruning(net, valid_dataloader, criterion, optimizer, update=False)
            if valid_acc > now_best_acc:
                now_best_acc = valid_acc
                # sava the best model weights each width_mult
                torch.save(net.state_dict(), f'custom_small_rate_{now_width_mult}.bin')
            print('rate %6.4f epoch %3d: train loss %6.4f, acc: %6.4f valid loss: %6.4f, acc: %6.4f'
                  % (now_width_mult, epoch, train_loss, train_acc, valid_loss, valid_acc))