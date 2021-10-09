import json
import os

import numpy
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from prepropess import Preprocess
from data import SentenceDataset
from model import Seq2Seq, Encoder, Decoder
from train import train_loop, valid_loop
from config import configurations
from utils import load_model, save_model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prepropess = Preprocess(cmn_eng_path='./cmn-eng')
    config = configurations()
    en_vocab_size = len(prepropess.word2int_en.keys())
    cn_vocab_size = len(prepropess.word2int_cn.keys())
    train_dataset = SentenceDataset(prepropess,
                                    max_len=config.max_output_len,
                                    set_name='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True)
    valid_dataset = SentenceDataset(prepropess,
                                    max_len=config.max_output_len,
                                    set_name='valid')
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=1)

    encoder = Encoder(en_vocab_size,
                      config.emb_dim,
                      config.hid_dim,
                      config.n_layers,
                      config.dropout)
    decoder = Decoder(cn_vocab_size,
                      config.emb_dim,
                      config.hid_dim,
                      config.n_layers,
                      config.dropout,
                      config.attention)
    model = Seq2Seq(encoder, decoder, device)

    opt = torch.optim.Adam(model.parameters(),
                           lr=config.learning_rate)
    if config.load_model:
        model = load_model(model,
                           config.load_model_path)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []
    step = 0
    while step < config.num_steps:
        loss = train_loop(model,
                          device,
                          opt,
                          loss_fn,
                          train_dataloader,
                          config.summary_steps,
                          step)
        train_losses += loss
        val_loss, bleu_score, result = valid_loop(model,
                                                  device,
                                                  loss_fn,
                                                  valid_dataloader)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        step += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}"\
               .format(step, val_loss, numpy.exp(val_loss), bleu_score))

    save_model(model, opt, config.store_model_path, config.num_steps)
