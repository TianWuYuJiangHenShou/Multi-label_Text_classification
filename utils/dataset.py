#_*_coding:utf-8_*_

from torch.utils import data
import torch as t
import numpy as np
import random
from glob import glob
import json


class ZhihuData(data.Dataset):
    '''
    主要用到的数据集
    '''

    def __init__(self, train_root, labels_file, type_='char', augument=True):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)

        # embedding_d = np.load(embedding_root)['vector']
        self.augument = augument
        question_d = np.load(train_root)
        self.type_ = type_
        if type_ == 'char':
            all_data_title, all_data_content = \
                question_d['title_char'], question_d['content_char']

        elif type_ == 'word':
            all_data_title, all_data_content = \
                question_d['title_word'], question_d['content_word']

        self.train_data = all_data_title[:-200000], all_data_content[:-200000]
        self.val_data = all_data_title[-200000:], all_data_content[-200000:]

        self.all_num = len(all_data_content)
        # del all_data_title,all_data_content

        self.data_title, self.data_content = self.train_data
        self.len_ = len(self.data_title)

        self.index2qid = question_d['index2qid'].item()
        self.l_end = 0
        self.labels = labels_['d']

        self.training = True

    def shuffle(self, d):
        return np.random.permutation(d.tolist())

    def dropout(self, d, p=0.5):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 0
        return d

    def train(self, train=True):
        if train:
            self.training = True
            self.data_title, self.data_content = self.train_data
            self.l_end = 0
        else:
            self.training = False
            self.data_title, self.data_content = self.val_data
            self.l_end = self.all_num - 200000
        self.len_ = len(self.data_content)
        return self

    def __getitem__(self, index):
        title, content = self.data_title[index], self.data_content[index]

        if self.training and self.augument:
            augument = random.random()

            if augument > 0.5:
                title = self.dropout(title, p=0.3)
                content = self.dropout(content, p=0.7)
            else:
                title = self.shuffle(title)
                content = self.shuffle(content)

        qid = self.index2qid[index + self.l_end]
        labels = self.labels[qid]
        data = (t.from_numpy(title).long(), t.from_numpy(content).long())
        label_tensor = t.zeros(1999).scatter_(0, t.LongTensor(labels), 1).long()
        return data, label_tensor

    def __len__(self):
        return self.len_