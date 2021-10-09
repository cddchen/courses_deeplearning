import random

import torch
import math
from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                 en_vocab_size,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(en_vocab_size,
                                      emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim,
                          hid_dim,
                          n_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch_size, seq_len, vocab_size]
        embedding = self.embedding(input)
        # embedding = [batch_size, seq_len, emb_dim]
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [seq_len, batch, num_directions * hid_dim]
        # hidden = [n_layers * directions, batch, hid_dim]

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self,
                 cn_vocab_size,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 dropout,
                 isatt=False):
        super(Decoder, self).__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        # self.attention = Attention(hid_dim)
        self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim,
                          self.hid_dim,
                          self.n_layers,
                          dropout=dropout,
                          batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim,
                                          self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2,
                                          self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4,
                                          self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size, vocab_size]
        # hidden = [batch_size, n_layers * directions, hid_dim]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch_size, 1, emb_dim]
        # if self.isatt:
            # attn = self.attention(encoder_outputs, hidden)
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch_size, 1, hid_dim]
        # hidden = [num_layers, batch_size, hid_dim]

        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch_size, vocab_size]
        return prediction, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch_size, seq_len, hid_dim * directions]
        # decoder_hidden = [num_layers, batch_size, hid_dim]
        # 一般取encoder最后一层的hidden state来做attention
        attention = None

        return attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input, target, teacher_forcing_ratio):
        # input = [batch_size, input_len, vocab_size]
        # target = [batch_size, target_len, vocab_size]
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        outputs = torch.zeros(batch_size,
                              target_len,
                              vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input)
        # Encoder最后隐藏层(hidden state)用来初始化Decoder
        # encoder_outputs主要是使用在Attention
        # 因为Encoder是双向的RNN，所以需要将同一层两个方向的hidden state接在一起
        # hidden = [num_layers * directions, batch_size, hid_dim] -> [num_layers, directions, batch_size, hid_dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :],
                            hidden[:, -1, :, :]), dim=2)
        # 取的<BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input,
                                           hidden,
                                           encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正确答案来做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出概率最大的单词
            top1 = output.argmax(1)
            # 如果是 teacher_force 则用正解训练，反之用自己预测的答案训练
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)

        return outputs, preds

    def inference(self, input, target):
        # 这里实施 Beam Search
        # 此函数 batch_size = 1
        # input = [batch_size, input_len, vocab_size]
        # target = [batch_size, target_len, vocab_size]
        batch_size = input.shape[0]
        input_len = input.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        outputs = torch.zeros(batch_size,
                              input_len,
                              vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input)
        # Encoder's hidden state to init Decoder
        # encoder_outputs use to do attention
        # Encoder direction is bidirectional, so do cat bidirectional hidden state to one
        # hidden = [num_layers * directions, batch_size, hid_dim] -> [num_layers, directions, batch_size, hid_dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :],
                            hidden[:, -1, :, :]), dim=2)
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input,
                                          hidden,
                                          encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding From Formula
        pos_encoding = torch.zeros(max_len, dim_model)
        # 0, 1, 2, 3, 4, 5, ...
        position_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        # 1000^(2i/dim_model)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])



