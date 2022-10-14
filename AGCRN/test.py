# _*_ coding: utf-8 _*_
# @time     :2022/3/1 15:22
# @Author   :jc
# @File     :test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

data1 = np.load("data/PEMSD8/pems08.npz")['data'][:, :, 0]
# data2 = np.load("data/PEMSD4/pems04.npz")['data'][:, :, 0]
# # np.savetxt("pems04.csv", data, delimiter=",")
# print(data2.shape)

exit()

from AGCRN.model.EmbGCN import Spatial_Attention_layer
# sa=Spatial_Attention_layer(170,3,3)
# a=torch.randn(64,170,3)
# res=sa(a)
# print(a.shape)

SVR_MAPE = [12.27,13.46,14.69,15.97,16.96,18.64,19.57,20.83,21.18,23.66,25.37,26.73]
SVR_RMSE = [29.95,32.65,35.31,37.81,40.15,42.77,45.08,47.5,50,53.89,57.2,62]
sum=0
for i in SVR_RMSE:
    sum=sum+i
print(sum/12)
exit()


#STFGNN  D8
# [125, 14.33176861378684, 16.726921822312235, 16.704094070142812, 10.571109083770445, 26.265921894063844,
#  [(13.978338178957296, 9.02985477894324, 21.495462437654542),
#   (14.323117001787935, 9.219252233420743, 22.122311487329874),
#   (14.632908961895158, 9.407911876661268, 22.684952293060274),
#   (14.90297142694463, 9.543783734756143, 23.177855794084003),
#   (15.16111016703254, 9.679805741123035, 23.63726722199391),
#   (15.406069389728895, 9.808732373392095, 24.072903425043087),
#   (15.637384128302733, 9.93921354247021, 24.476439739332204),
#   (15.858330324743045, 10.072465822276715, 24.852847748610934),
#   (16.07086989405664, 10.19188407404432, 25.21003052906501),
#   (16.276214688529734, 10.312511525547697, 25.556031195275132),
#   (16.483281696659414, 10.43669178326803, 25.903012028688483),
#   (16.704094070142812, 10.571109083770445, 26.265921894063844)]]

#STFGNN  D4
# [169, 17.016524635770786, 19.578628614381312, 19.921787663778776, 13.098494453969508, 32.20772161579708,
res_d4=[(17.604829375394345, 11.827138109995628, 28.287051615014377),
(17.935521601307762, 11.999429621183308, 28.847090148239285),
(18.22449785748794, 12.162165796457712, 29.341131682305775),
(18.468251734874332, 12.29162469703126, 29.764515858525947),
(18.68615452167849, 12.407342872556798, 30.14715265133409),
(18.883357973542996, 12.511840650575373, 30.491537777919934),
(19.075205487707535, 12.612906906673638, 30.819102260049757),
(19.260107612721928, 12.710408744434835, 31.1309299240876),
(19.433378997045924, 12.808849553410587, 31.42106735636841),
(19.594550817651275, 12.902083432074768, 31.689892581809133),
(19.752710107959995, 12.995018968682048, 31.946222828060126),
(19.921787663778776, 13.098494453969508, 32.20772161579708)]


res_d8=[(13.978338178957296, 9.02985477894324, 21.495462437654542),
  (14.323117001787935, 9.219252233420743, 22.122311487329874),
  (14.632908961895158, 9.407911876661268, 22.684952293060274),
  (14.90297142694463, 9.543783734756143, 23.177855794084003),
  (15.16111016703254, 9.679805741123035, 23.63726722199391),
  (15.406069389728895, 9.808732373392095, 24.072903425043087),
  (15.637384128302733, 9.93921354247021, 24.476439739332204),
  (15.858330324743045, 10.072465822276715, 24.852847748610934),
  (16.07086989405664, 10.19188407404432, 25.21003052906501),
  (16.276214688529734, 10.312511525547697, 25.556031195275132),
  (16.483281696659414, 10.43669178326803, 25.903012028688483),
  (16.704094070142812, 10.571109083770445, 26.265921894063844)]
mae_d4=0
mape_d4=0
rmse_d4=0
mae_d8=0
mape_d8=0
rmse_d8=0
for i in range(12):
    mae_d4+=res_d4[i][0]
    mape_d4+=res_d4[i][1]
    rmse_d4+=res_d4[i][2]
    mae_d8 += res_d8[i][0]
    mape_d8 += res_d8[i][1]
    rmse_d8 += res_d8[i][2]

print("d4: mae:",mae_d4/12,' rmse:',mape_d4/12,' mape:',rmse_d4/12)
print("d8: mae:",mae_d8/12,' rmse:',mape_d8/12,' mape:',rmse_d8/12)

exit()

data=csv.reader(open(r'G:\研究方向论文\读\Data\PeMSD7_V_228.csv','r'))
# data_npz=np.load(r'G:\研究方向论文\读\已读\AGCRN\AGCRN-master\traffic_predict\AGCRN\data\PEMSD7\PEMS07.npz')
# print(data_npz['data'].shape)
# print(type(data_npz['data'][0,0,0]))
sum=0
data_npz=[]
for line in data:
    r = []
    for i in range(len(line)):
        r.append(float(line[i]))
    print(len(r))
    data_npz.append(r)
    print(data_npz)
    # print(type(line[0]))
    # print(len(line))
    exit()
print(sum)





# aa=torch.randn((4,4,2))
# bb=torch.einsum('bnc,bcm->bnm',aa,aa.permute(0,2,1))
# print(bb)
# dd=[]
# for i in range(4):
#     cc=torch.einsum('nc,cm->nm',aa[i,:,:],aa[i,:,:].permute(1,0))
#     dd.append(cc)
#
# res=torch.cat(dd,dim=0)
# print(res)

exit()


# AGCRN_mae  = [18.8,18.95,19.11,19.28,19.41,19.66,19.84,20.08,20.27,20.42,20.81,21.44]
# AGCRN_rmse = [29.83,30.22,30.72,31.24,31.72,31.99,32.56,32.98,33.32,33.74,34.12,34.97]
# AGCRN_mape = [12.12,12.18,12.35,12.48,12.54,12.69,12.94,13.17,13.32,13.48,13.76,14.12]
# Ours_model_mae=[18.05,18.18,18.45,18.71,18.92,19.08,19.26,19.58,19.8,19.95,20.16,20.55]
# Ours_model_rmse=[29.62,30.09,30.62,31.25,31.67,31.84,32.28,32.78,33.18,33.48,33.86,34.38]
# Ours_model_mape=[12.04,12.11,12.3,12.45,12.52,12.64,12.82,13.01,13.14,13.22,13.37,13.6]
# sum=0
# for i in Ours_model_mae:
#     sum+=i
# print(sum/12)
# sum=0
# for i in Ours_model_rmse:
#     sum+=i
# print(sum/12)
# sum=0
# for i in Ours_model_mape:
#     sum+=i
# print(sum/12)
# exit()


net=torch.load(r'C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD8\3.1\qk_conv2d\best_model.pth', map_location=lambda storage, loc: storage)
print(net)
# for i in net.values:
#     print(i)
nb=net['node_embeddings']

adj=torch.mm(nb,nb.transpose(0,1))
# print(max(adj))
a = np.random.rand(4,3)
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(adj,
                annot=False, vmax=1,vmin = -1, xticklabels= False, yticklabels= False, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
#            square=True, cmap="YlGnBu")
# ax.set_title('二维数组热力图', fontsize = 18)
# ax.set_ylabel('数字', fontsize = 18)
# ax.set_xlabel('字母', fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
# ax.set_yticklabels(['一', '二', '三'], fontsize = 18, rotation = 360, horizontalalignment='right')
# ax.set_xticklabels(['a', 'b', 'c'], fontsize = 18, horizontalalignment='right')
plt.show()






exit()

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None,score_his=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)
    # if score_his != None:
    #     scores=scores+score_his
    #     score_his=scores
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(32, d_model), 4)
        self.lin=nn.Linear(32,64)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, score_his=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 这里是将前面定义的多个linear与query key value分别取出然后经过线性层变换然后转换维度得到query key value
        # query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
        #                      zip(self.linears, (query, key, value))]
        # query=self.lin(query)
        # key=self.lin(key)
        # value=self.lin(key)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        # 下面是res-attention
        # b,n,t1,_=query.shape
        # h_s=self.h
        # t2=value.shape[2]
        # score_his=torch.zeros((b,n,h_s,t1,t2)) # scores: (batch, N, h, T1, T2)
        # x, self.attn,score_his = res_attention(query, key, value,score_his, mask=mask, dropout=self.dropout)
        print(x.shape)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        # return self.linears[-1](x)
        return x

def aa():
    node_embeddings = nn.Parameter(torch.randn(170,3), requires_grad=True)
    print(node_embeddings.shape)
    supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
    print(supports.shape)

def load_parameters(file_path):
    #G:\研究方向论文\读\已读\AGCRN\AGCRN-master\traffic_predict\AGCRN\model\model_para\PEMSD7
    state_dict = torch.load(file_path)
    for i in state_dict:
        print(i)
if __name__=="__main__":
    # x=torch.randn((64,12,10,32))
    # mha=MultiHeadAttention(8,64)
    # out=mha(x,x,x)
    # aa()
    file_path=r"G:\研究方向论文\读\已读\AGCRN\AGCRN-master\traffic_predict\AGCRN\model\model_para\PEMSD7\epoch_75.pth"
    load_parameters(file_path)

    
    # import numpy as np
    #
    # path=r'G:\研究方向论文\读\Data\Data\METR-LA\test.npz'
    # test_data=np.load(path)
    # print(test_data.files)
    # print(test_data['x'].shape)
    # print(test_data['y'].shape)
    # print(test_data['x_offsets'].shape)

    # aa=torch.cat([a[0],a[5]],2)
    # print(aa.shape)
    # # print(a[5].shape)
