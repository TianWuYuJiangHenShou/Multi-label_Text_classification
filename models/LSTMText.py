#_*_coding:utf-8_*_

import torch
import numpy as np
from torch import nn
from .BasicModule import BasicModule


def kmax_pooling(x,dim,k):
    index = x.topk(k,dim= dim)[1].sort(dim=dim)[0]
    return x.gather(dim,index)

class LSTMText(BasicModule):

    def __init__(self,opt):
        super(LSTMText,self).__init__()
        self.model_name = 'LSTMText'
        self.opt = opt

        kernel_size = opt.kernel_size
        self.embed = nn.Embedding(opt.vocab_size,opt.embedding_dim)#size(行，列)
        self.title_lstm = nn.LSTM(input_size=opt.embedding_dim,
                                  hidden_size=opt.hidden_size,
                                  num_layers=self.num_layers,
                                  bias=True,
                                  batch_first=False,
                                  bidirectional=True)
        self.content_lstm = nn.LSTM(input_size=opt.embedding_dim,
                                    hidden_size=opt.hidden_size,
                                    num_layers=self.num_layers,
                                    bias=True,
                                    batch_first=False,
                                    bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling * (opt.hidden_size *2 *2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

        if opt.embedding_path:
            self.embed.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title,content):
        title = self.embed(title)  #(seq,batch,imput_size)
        content = self.embed(content)

        if self.opt.static:
            title = title.detach()
            content = content.detach()

        title_out = self.title_lstm(title.permute(1,0,2))[0].permute(1,2,0)#(batch,hidden,seq)
        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)#(batch,hidden,seq)

        title_conv_out = kmax_pooling(title_out,2,self.kmax_pooling) #(batch,hidden,kmax_pooling)
        content_conv_out = kmax_pooling(content_out,2,self.kmax_pooling) #(batch,hidden,kmax_pooling)

        conv_out = torch.cat((title_conv_out,content_conv_out),dim=1)  #(batch,hidden * 2,kmax_pooling)
        reshaped = conv_out.view(conv_out.size(0),-1)   #(batch,hidden * 2 * kmax_pooling)
        '''
        reshaped是LSTM层输出的reshaped,size:(batch,hidden * 2 * kmax_pooling) 
        而下一层线性层的输入size:(hidden * 2 * kmax_pooling * 2,linear_hidden_size)
        按理说reshaped的第二维与linear的第一维是相等的，这里不一样是什么原因？
        因为这里LSTM层采用的是Bi-LSTM(双向LSTM)，最终会torch.cat((正向LSTM的输出，反向LSTM的输出),1)，这样在维度上就等了
        '''
        logits = self.fc((reshaped))

        return logits





