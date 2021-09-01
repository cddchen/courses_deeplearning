import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_sz, embed_dim):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, 1)
        self.logistic = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        fced = self.fc(embedded)
        return self.logistic(fced)


class CustomDataSet(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]


def process_data(path, is_label):
    data_, label = [], []
    with open(path, "r") as f:
        for row in f.readlines():
            if is_label:
                row = row.split(' +++$+++ ')
                label.append(int(row[0].strip()))
                data_.append(row[1].strip())
            else:
                data_.append(row.strip())
    if is_label:
        return data_, label
    else:
        return data_


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int32)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.float32)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def no_label_collate_batch(batch):
    text_list, offsets =  [], [0]
    for _text in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int32)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(device), offsets.to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        predicted_label = predicted_label.flatten()
        loss = criterion(predicted_label, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (torch.round(predicted_label) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            predicted_label = predicted_label.flatten()
            loss = criterion(predicted_label, label)
            total_acc += (torch.round(predicted_label) == label).sum().item()
            total_count += label.size(0)
        return total_acc/total_count


def read_label_file(file):
    with open(file, 'r') as f:
        ret = []
        for _ in f.readlines():
            ret.append(float(_))
    return ret


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # process data
    print("*" * 59)
    print("device is {}".format(device))
    print("*" * 59)
    data, label_ = process_data('./training_label.txt', True)
    no_label_data = process_data('./training_nolabel.txt', False)
    label__ = read_label_file('./label.txt')
    data = np.concatenate((data, no_label_data), axis=0)
    label_ = np.concatenate((label_, label__), axis=0)
    train_dataset = CustomDataSet(data, label_)
    # define some function to process token
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    # train
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize).to(device)
    EPOCHS = 10
    LR = 5
    BATCH_SIZE = 64
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    num_train = int(len(train_dataset) * 0.95)
    split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)
        path = 'model.pth'
        torch.save(model.state_dict(), path)