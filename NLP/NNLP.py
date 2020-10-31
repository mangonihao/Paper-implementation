# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 13:49
# @Author  : XMH
# @File    : NNLP.py
# @Software: PyCharm
# @Description:
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import argparse
import numpy as np
class NNLP(nn.Module):
    def __init__(self,embedding_dim, n_grams_dim,hidden_dim, vocab_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim
        self.n_grams_dim = n_grams_dim

        self.C = nn.Embedding(self.vocab_dim, self.embedding_dim)
        self.H = nn.Linear((self.n_grams_dim-1) * self.embedding_dim, self.hidden_dim)
        self.U = nn.Linear(self.hidden_dim, self.vocab_dim)
        self.W = nn.Linear((self.n_grams_dim-1)*self.embedding_dim, self.vocab_dim, bias=False)
        self.b = nn.Parameter(torch.ones(self.vocab_dim))

    def forward(self,x):
        x = self.C(x) #[batch_size, (n_grams_dim-1), embedding_dim]
        x = x.view(-1,(self.n_grams_dim-1)*self.embedding_dim) #[batch_size, (n_grams_dim-1)* embedding_dim]
        out = torch.tanh(self.H(x))
        out = self.U(out)
        out1 = out + self.W(x) + self.b
        # print("nihao")
        # print(out1.shape) #[batch_size, vocab_dim]
        return out1

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_dim',type=int)
    parser.add_argument('n_grams_dim',type=int)
    parser.add_argument('hidden_dim',type=int)
    # parser.add_argument('batch_size',type=int)
    parser.add_argument('epochs',type=int)
    parser.add_argument('lr',type=float)

    args = parser.parse_args()
    args_map = vars(args)
    return args_map

def make_dict(text): #制作词典
    word_list = " ".join(text).split()
    vocab_dim = len(word_list)
    word2id = {w:i for i,w in enumerate(word_list)}
    id2word = {i:w for i,w in enumerate(word_list)}

    return vocab_dim, word2id, id2word

def make_data(text, word2id, n_grams_dim):
    input_batch = []
    target_batch = []
    for sentence in text:
        sentence = sentence.split()
        if len(sentence) < n_grams_dim: #一个补丁
            continue
        for i in range(0,len(sentence)):
            if i + n_grams_dim > len(sentence):
                continue
            n_grams_content = sentence[i:i+n_grams_dim]
            input_batch.append([word2id[x] for x in n_grams_content[:-1]])
            target_batch.append(word2id[n_grams_content[-1]])
    return input_batch, target_batch

class NNLP_Tools():
    def __init__(self,args,vocab_dim):
        self.args = args #保存一些超参数
        self.vocab_dim = vocab_dim
        self.nnlp = NNLP(self.args['embedding_dim'], self.args['n_grams_dim'], self.args['hidden_dim'],vocab_dim)
        self.opt = optim.Adam(self.nnlp.parameters(),lr = self.args['lr'])
        # self.loss_func = nn.CrossEntropyLoss
        self.loss_func = F.cross_entropy
    def loss_batch(self,xb,yb,opt=None):
        pred = self.nnlp(xb)
        print(yb.shape)
        losses = self.loss_func(pred,yb)

        if opt is not None:
            opt.zero_grad()
            losses.backward()
            opt.step()
        return losses, len(xb)

    def train(self,x_data,y_data):
        # print(x_data.shape)
        self.nnlp.train()
        for epoch in range(args['epochs']):
            loss = self.loss_batch(x_data,y_data,self.opt)
            if (epoch+1) % 4 == 0:
                print("Epoch {0}: {1}".format(epoch,loss))

        # self.nnlp.eval()
        # with torch.no_grad():
        #     losses, numes = zip(*[self.loss_batch(x_data,y_data)])

    def test(self,x_data):
        pred = self.nnlp(x_data).data.max(1,keepdim = True)[1]
        return pred



if __name__ == '__main__':
    '''example 
    >>> python NNLP.py 8 4 8 8 0.1
    '''

    args = arg_parse() #获取超参数

    text = ['I love you', 'you are a beautiful woman', 'He likes bread']
    print(args)

    vocab_dim, word2id, id2word = make_dict(text)

    input_batch, target_batch = make_data(text, word2id, args['n_grams_dim'])
    print(input_batch)
    print(target_batch)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    nnlp_model = NNLP_Tools(args,vocab_dim)
    nnlp_model.train(input_batch,target_batch)
    pred = nnlp_model.test(input_batch)
    pred_list = np.squeeze(pred.numpy())
    for x, y in zip(input_batch.data.numpy(), pred_list):
        for id in x:
            print(id2word[id],end=' ')
        print('->',end=' ')
        print(id2word[y])










