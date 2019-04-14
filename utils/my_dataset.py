#_*_coding:utf-8_*_

from torch.utils import data
import torch
import json
import numpy as np
import random
import fire

#只针对train、test

class ZhihuData(data.Dataset):

    def __init__(self,data_path,label_path,type_ = 'char',train = True,augument = False):

        with open(label_path) as f:
            labels_ = json.load(f)

        self.augument = augument
        self.type_ = type_
        data = np.load(data_path)
        if type_ == 'char':
            title,content = data['title_char'],data['content_char']
        elif type_ == 'word':
            title,content = data['title_word'],data['content_word']

        self.all_num = len(content)

        if train:
            self.train_data = title[:-200000], content[:-200000]
            self.title,self.content = self.train_data
            self.l_end = 0
        else:
            self.val_data = title[-200000:],content[-200000:]
            self.title,self.content = self.val_data
            self.l_end = self.all_num - 200000


        self.len_ = len(self.title)

        self.index2id = data['index2qid'].item()
        self.labels = labels_['d']    #d:{qid:list(label)}
        self.training = True

    def shuffle(self, d):
        return np.random.permutation(d.tolist())

    def dropout(self, d, p=0.5):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 0
        return d

    def __getitem__(self, index):  #针对一行的对label或者train_data 的操作放在此
        title_,content_ = self.title[index],self.content[index]

        if self.training and self.augument:
            augument = random.random()

            if augument > 0.5:
                title_ = self.dropout(title_,p = 0.3)
                content_ = self.dropout(content_,p = 0.7)
            else:
                title_ = self.shuffle(title_)
                content_ = self.shuffle(content_)

        qid = self.index2id[index+self.l_end]
        labels = self.labels[qid]   #list
        input = (torch.from_numpy(title_).long(),torch.from_numpy(content_).long())
        label_tensor = torch.zeros(1999).scatter_(0, torch.LongTensor(labels), 1).long()
        return input,label_tensor

    def __len__(self):
        return self.len_

if __name__ == "__main__":
    fire.Fire()