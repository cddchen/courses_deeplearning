import numpy as np
import torch
import matplotlib.pyplot as plt
from units import readfile
from models import Attacker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, y, categorys = readfile('./data/images', './data/labels.csv', './data/categories.csv')
attacker = Attacker(x, y)
epsilons = [0.1, 0.01]

accuracies, examples = [], []

for eps in epsilons:
  ex, acc = attacker.attack(eps)
  accuracies.append(acc)
  examples.append(ex)

cnt = 0
plt.figure(figsize=(30, 30))
for i in range(len(epsilons)):
  for j in range(len(examples[i])):
    cnt += 1
    plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    if j == 0:
        plt.ylabel("Eps: %f" % (epsilons[i]), fontsize=14)
    orig, adv, orig_img, ex = examples[i][j]
    plt.title('original: %s' % (categorys[orig].split(',')[0]))
    orig_img = np.transpose(orig_img, (1, 2, 0))
    plt.imshow(orig_img)
    cnt += 1
    plt.subplot(len(epsilons), len(examples[0]) * 2, cnt)
    plt.title('adversarial: %s' % (categorys[adv].split(',')[0]))
    ex = np.transpose(ex, (1, 2, 0))
    plt.imshow(ex)
plt.tight_layout()
plt.show()