import torch
import numpy as np
from utils import schedule_sampling,\
    tokens2sentence, computebleu


def train_loop(model, device, opt, loss_fn, dataloader, summary_steps, total_steps) -> list:
    model.train()
    model.zero_grad()
    total_loss = 0
    losses = []

    for step in range(summary_steps):
        sources, targets = next(iter(dataloader))
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling())
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_fn(outputs, targets)

        opt.zero_grad()
        loss.backward()
        gard_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

        total_loss += loss.item()
        if (step + 1) % 5 == 0:
            total_loss = total_loss / 5
            print('\r', 'train [{}] loss: {:.3f}, Perplexity: {:.3f}'\
                  .format(total_steps + step + 1, total_loss, np.exp(total_loss)), end=' ')
            losses.append(total_loss)
            total_loss = 0

    return losses


def valid_loop(model, device, loss_fn, dataloader):
    model.eval()
    total_loss, bleu_score = 0.0, 0.0
    n = 0
    result = []

    for batch in dataloader:
        sources, targets = batch
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds,
                                dataloader.dataset.preproposs.int2word_cn)
        sources = tokens2sentence(sources,
                                  dataloader.dataset.preproposs.int2word_en)
        targets = tokens2sentence(targets,
                                  dataloader.dataset.preproposs.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))

        bleu_score += computebleu(preds, targets)
        n += batch_size

    return total_loss / len(dataloader), bleu_score / n, result
