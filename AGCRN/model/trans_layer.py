# _*_ coding: utf-8 _*_
# @time     :2021/12/30 12:08
# @Author   :jc
# @File     :trans_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

device=torch.device('cuda')
# device=torch.device('cpu')

class Transform2(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        # self.qff = nn.Linear(outfea, outfea)
        # # nn.init.kaiming_uniform_(self.qff.weight,nonlinearity="relu")
        # self.kff = nn.Linear(outfea, outfea)
        # nn.init.kaiming_uniform_(self.kff.weight, nonlinearity="relu")
        # self.vff = nn.Linear(outfea, outfea)
        # nn.init.kaiming_uniform_(self.vff.weight, nonlinearity="relu")
        self.conv1=nn.Conv2d(12,12,(1,3),bias=True)
        self.conv2=nn.Conv2d(12,12,(1,3),bias=True)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x,score_his=None):# x : b t n hidden
        b, t, n, c = x.shape
        # query = self.qff(x)
        # key = self.kff(x)
        # value = self.vff(x)
        query=self.conv1(x)
        key=self.conv2(x)
        value=self.vff(x)
        # query=x
        # key=x
        # value=x
        query = query.permute(0, 2, 1, 3)
        # print(query.shape)
        key = key.permute(0, 2, 3, 1)
        # print(key.shape)
        value = value.permute(0, 2, 1, 3)

        # query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        # # print(query.shape)
        # key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        # # print(key.shape)
        # value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)



        A = torch.matmul(query, key)
        # print("A:",A.shape)
        A /= (c ** 0.5)
        # print(A.shape,"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaaa")

        # print(score_his.shape)

        A = torch.softmax(A, -1)

        # 残差注意力
        if score_his is not None:
            try:
                A=A+score_his
            except:
                pass
        score_his=A.clone().detach()

        value = torch.matmul(A, value)
        # value = torch.softmax(value,-2)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        # value = value.permute(0,2,1,3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x),score_his
class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        # self.qff = nn.Linear(outfea, outfea)
        # # nn.init.kaiming_uniform_(self.qff.weight,nonlinearity="relu")
        # self.kff = nn.Linear(outfea, outfea)
        # nn.init.kaiming_uniform_(self.kff.weight, nonlinearity="relu")
        self.vff = nn.Linear(outfea, outfea)
        # nn.init.kaiming_uniform_(self.vff.weight, nonlinearity="relu")
        self.conv1=nn.Conv2d(12,12,(1,3),bias=True)
        self.conv2=nn.Conv2d(12,12,(1,3),bias=True)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)
        # self.ln = nn.LayerNorm(12)
        # self.lnff = nn.LayerNorm(12)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = 2

    def forward(self, x,score_his=None):# x : b t n hidden
        b, t, n, c = x.shape
        # query = self.qff(x)
        # key = self.kff(x)
        # value = self.vff(x)
        query=self.conv1(x)
        # print(query.shape)
        key=self.conv2(x)
        value=self.vff(x)
        query = query.permute(0, 2, 1, 3)
        # print(query.shape)
        key = key.permute(0, 2, 3, 1)
        # print(key.shape)
        value = value.permute(0, 2, 1, 3)

        # query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        # # print(query.shape)
        # key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        # # print(key.shape)
        # value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)



        A = torch.matmul(query, key)
        # print("A:",A.shape)
        A /= (c ** 0.5)

        # print(score_his.shape)

        A = torch.softmax(A, -1)

        # 残差注意力
        if score_his is not None:
            try:
                A=A+score_his
            except:
                pass
        score_his=A.clone().detach()

        value = torch.matmul(A, value)
        # value = torch.softmax(value,-2)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        # value = value.permute(0,2,1,3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x),score_his


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=12):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe,
                         requires_grad=False)
        return x

class transformer_layer(nn.Module):
    def __init__(self,dim_in,dim_out,num_layer,d=2,att_his=False):
        super(transformer_layer,self).__init__()
        # self.linear1=nn.Linear(dim_in,dim_out)
        self.trans_layers=nn.ModuleList(Transform(dim_out,d) for l in range(num_layer))
        self.PE=PositionalEncoding(dim_out)
        self.num_layer=num_layer
        self.att_his=att_his
        self.score_his = torch.zeros((64, 170, 12, 12), requires_grad=False).to(device)
    def forward(self, x):
        # x=self.linear1(x)
        x=self.PE(x)
        # print(x.shape,"xxxx")
        for l in range(self.num_layer):
            if  self.att_his:
                x,self.score_his=self.trans_layers[l](x,self.score_his)
            else:
                x,_=self.trans_layers[l](x)
        return x

class PositionalEncoding_S(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=170):
        super(PositionalEncoding_S, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe,
                         requires_grad=False)
        return x
class transformer_layer_S(nn.Module):
    def __init__(self,dim_in,dim_out,num_layer,d,att_his=True):
        super(transformer_layer_S,self).__init__()
        # self.linear1=nn.Linear(dim_in,dim_out)
        self.trans_layers=nn.ModuleList(Transform(dim_out,d) for l in range(num_layer))
        self.PE=PositionalEncoding_S(dim_out)
        self.num_layer=num_layer
        self.att_his=att_his
        self.score_his = torch.zeros((64, 170, 12, 12), requires_grad=False).to(device)
    def forward(self, x):
        # x=self.linear1(x)
        x=self.PE(x)
        for l in range(self.num_layer):
            if  self.att_his:
                x,self.score_his=self.trans_layers[l](x,self.score_his)
            else:
                x,_=self.trans_layers[l](x)
        return x



if __name__=="__main__":
    x = torch.randn(32, 12, 170, 64)
    dim_in=64
    dim_out=64
    num_layer=2
    d=64
    transformer=transformer_layer(dim_in,dim_out,num_layer,d)
    res=transformer(x)
    print(res.shape)