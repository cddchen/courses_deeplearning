import json
import os.path
import torch
import numpy as np


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


class Preprocess():
    def __init__(self, cmn_eng_path):
        with open(os.path.join(cmn_eng_path, 'int2word_cn.json'), encoding='UTF-8') as f:
            self.int2word_cn = json.load(f)
        with open(os.path.join(cmn_eng_path, 'int2word_en.json'), encoding='UTF-8') as f:
            self.int2word_en = json.load(f)
        with open(os.path.join(cmn_eng_path, 'word2int_cn.json'), encoding='UTF-8') as f:
            self.word2int_cn = json.load(f)
        with open(os.path.join(cmn_eng_path, 'word2int_en.json'), encoding='UTF-8') as f:
            self.word2int_en = json.load(f)
        self.train_en = []
        self.train_cn = []
        self.valid_en = []
        self.valid_cn = []
        self.test_en = []
        self.test_cn = []
        with open(os.path.join(cmn_eng_path, 'training.txt'), encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                sen_en, sen_cn = line[0], line[1]
                self.train_en.append(sen_en)
                self.train_cn.append(sen_cn)
        with open(os.path.join(cmn_eng_path, 'validation.txt'), encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                sen_en, sen_cn = line[0], line[1]
                self.valid_en.append(sen_en)
                self.valid_cn.append(sen_cn)
        with open(os.path.join(cmn_eng_path, 'testing.txt'), encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                sen_en, sen_cn = line[0], line[1]
                self.test_en.append(sen_en)
                self.test_cn.append(sen_cn)

    def word2vec_train(self):
        train_en = []
        for sen in self.train_en:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_en.keys():
                    tmp.append(self.word2int_en[word])
                else:
                    tmp.append(self.word2int_en["<UNK>"])
            train_en.append(tmp)
        train_cn = []
        for sen in self.train_cn:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_cn.keys():
                    tmp.append(self.word2int_cn[word])
                else:
                    tmp.append(self.word2int_cn["<UNK>"])
            train_cn.append(tmp)

        return train_en, train_cn

    def word2vec_valid(self):
        valid_en = []
        for sen in self.valid_en:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_en.keys():
                    tmp.append(self.word2int_en[word])
                else:
                    tmp.append(self.word2int_en["<UNK>"])
            valid_en.append(tmp)
        valid_cn = []
        for sen in self.valid_cn:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_cn.keys():
                    tmp.append(self.word2int_cn[word])
                else:
                    tmp.append(self.word2int_cn["<UNK>"])
            valid_cn.append(tmp)

        return valid_en, valid_cn

    def word2vec_test(self):
        test_en = []
        for sen in self.test_en:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_en.keys():
                    tmp.append(self.word2int_en[word])
                else:
                    tmp.append(self.word2int_en["<UNK>"])
            test_en.append(tmp)
        test_cn = []
        for sen in self.test_cn:
            sen = sen.split(' ')
            tmp = []
            for word in sen:
                if word in self.word2int_cn.keys():
                    tmp.append(self.word2int_cn[word])
                else:
                    tmp.append(self.word2int_cn["<UNK>"])
            test_cn.append(tmp)

        return test_en, test_cn
