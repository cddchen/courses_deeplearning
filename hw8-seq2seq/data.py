import numpy
import torch
from torch.utils import data
from prepropess import LabelTransform, Preprocess


class SentenceDataset(data.Dataset):
    def __init__(self, prepropess: Preprocess, max_len, set_name):
        self.preproposs = prepropess
        self.transform = LabelTransform(max_len, prepropess.word2int_en['<PAD>'])
        self.BOS = prepropess.word2int_en['<BOS>']
        self.EOS = prepropess.word2int_en['<EOS>']
        if set_name == 'train':
            self.en_sen, self.cn_sen = prepropess.word2vec_train()
        elif set_name == 'valid':
            self.en_sen, self.cn_sen = prepropess.word2vec_valid()
        elif set_name == 'test':
            self.en_sen, self.cn_sen = prepropess.word2vec_test()

    def __getitem__(self, idx):
        en_sen = [self.BOS] + self.en_sen[idx] + [self.EOS]
        en_sen = numpy.asarray(en_sen)
        cn_sen = [self.BOS] + self.cn_sen[idx] + [self.EOS]
        cn_sen = numpy.asarray(cn_sen)
        en_sen, cn_sen = self.transform(en_sen), self.transform(cn_sen)
        en_sen, cn_sen = torch.LongTensor(en_sen), torch.LongTensor(cn_sen)
        return en_sen, cn_sen

    def __len__(self):
        return len(self.en_sen)
