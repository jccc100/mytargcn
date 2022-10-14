import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


device=torch.device('cuda')
def sym_norm_Adj(W):

    W=W.to(device=torch.device('cpu'))
    assert W.shape[0] == W.shape[1]
    W=W.cpu().detach().numpy()

    N = W.shape[0]
    # D = np.zeros([N, N], dtype=type(W[0][0]))


    W = W + 0.5*np.identity(N) # 为邻居矩阵加上自连接

    D = np.diag(1.0/np.sum(W, axis=1))

    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)

    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix # D^-0.5AD^-0.5
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, num_node,c_in,c_out,dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        global device
        self.in_channels=c_in
        self.dropout = nn.Dropout(p=dropout)

        # self.conv1 = nn.Conv2d(num_node, num_node, (1, 3), bias=False)
        # self.conv2 = nn.Conv2d(num_node, num_node, (1, 3), bias=True)
        # #
        self.Wq=nn.Linear(c_in,c_out,bias=True)
        # nn.init.kaiming_uniform_(self.Wq.weight, nonlinearity="relu")
        self.Wk=nn.Linear(c_in,c_out,bias=True)
        # nn.init.kaiming_uniform_(self.Wk.weight, nonlinearity="relu")
        self.Wv=nn.Linear(c_in,c_out,bias=False)
        # # nn.init.kaiming_uniform_(self.Wv.weight, nonlinearity="relu")
    def forward(self, x,adj,score_his=None):
        '''
        :param x: (batch_size, N, C)
        :return: (batch_size, N, C)
        '''
        batch_size, num_of_vertices, in_channels = x.shape

        # Q K V 改之后
        Q=self.Wq(x)
        # Q=x
        # print("Q:",Q.shape)
        K=self.Wk(x)
        # K=x
        # print("K:", K.shape)
        V=self.Wv(x)
        # V=self.conv2(x)
        # V=x

        # Q=self.Wq(x)
        # Q=torch.split(Q,32,1)
        # K=self.Wk(x)
        # K=torch.split(K,32,1)
        # V=self.Wv(x)
        # V=torch.split(V,32,1)
        # sc=[]
        # for i in range(len(Q)):
        #     qk=torch.matmul(Q[i],K[i].permute(0,2,1))
        #     qk=torch.softmax(qk,dim=1)
        #     qkv=torch.matmul(qk,V[i])
        #     sc.append(qkv)
        # score = None
        # for i in range(len(sc) - 1):
        #     if score == None:
        #         score = torch.cat([sc[i], sc[i + 1]], 1)
        #     else:
        #         score= torch.cat([score, sc[i + 1]], 1)
        # score=torch.relu(score)
        # Q=torch.split(Q,10,)Q

        score = torch.matmul(Q, K.transpose(1, 2))
        score=F.softmax(score,dim=1)
        score=torch.einsum('bnm,mc->bnc',score,adj)
        score=torch.einsum("bnm,bmc->bnc",score,V)
        # score=torch.relu(score)
        # score=torch.einsum('bnn,bno->bno',score,V)#+x

        # score=torch.matmul(score,V)
        # score=F.relu(score)
        # # print("V:", V.shape)
        # if score_his!=None:
        #     score = torch.matmul(Q, K.transpose(1, 2))+score_his  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        # else:
        #     score = torch.matmul(Q, K.transpose(1, 2))
        # score=torch.softmax(score,dim=-1)
        # score=torch.matmul(score,V)


        # 改之前
        # if score_his!=None:
        #     score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)#+score_his  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        #     # score_his = score
        # else:
        #     score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)
        # #
        # # score=F.softmax(score,dim=-1)
        # score=torch.sigmoid(score+self.b_s) # b n n + 1 n n = b n n
        # # # score=torch.softmax(score+self.b_s,dim=-1) # b n n + 1 n n = b n n
        # # # score=torch.softmax(score,dim=1)
        # # # score=torch.einsum("nn,bnn->bnn",self.V_s,score)
        # score=torch.matmul(self.V_s,score)
        # score=torch.softmax(score,dim=1)



        #normalization
        # score=score-torch.max(score,1,keepdim=True)[0]
        # exp=torch.exp(score)
        # score_norm=exp/torch.sum(exp,1,keepdim=True)
        # score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b, N, N)

        # print(score_norm)
        # 公式6  返回注意力和更新的score_his用于下一次传参
        # return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)),score_his # (b t n n)
        return score # (b n n)
#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

class EmbGCN(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN, self).__init__()
        self.cheb_k = cheb_k
        # self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        # self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        # self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        # self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        # nn.parameter
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))


        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.normal_(self.weights_pool)
        nn.init.normal_(self.bias_pool)
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        support_set = [torch.eye(node_num).to(supports.device), supports]
        supports = torch.stack(support_set, dim=0)
        # 静态邻接矩阵
        # x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = self.linear(x_static)


        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv
        # return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))

        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
class EmbGCN2(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN2, self).__init__()
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        # self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports = torch.eye(node_num).to(supports.device)+supports
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        # 静态邻接矩阵
        x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        x_static = self.linear(x_static)
        #
        # x_static = self.SA(x,self.sym_norm_Adj_matrix)
        # x_static=F.relu(x_static)



        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, cheb_k, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
        # return x_gconv
        return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))

        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
class EmbGCN_linear(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_linear, self).__init__()
        self.cheb_k = cheb_k
        # self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        # self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        # self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        # self.linear_weight=nn.Parameter(torch.FloatTensor(170,cheb_k,dim_in,dim_out),requires_grad=True)
        # self.linear_b=nn.Parameter(torch.FloatTensor(170,dim_out),requires_grad=True)
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        supports=torch.eye(node_num).to(supports.device)+supports # 1+A
        # print("supports:",supports.shape) #2 170 170
        # 静态邻接矩阵
        # x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = self.linear(x_static)
        #
        # x_static = self.SA(x,self.sym_norm_Adj_matrix)
        # x_static=F.relu(x_static)



        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, N, dim_in

        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # TARGCN_linear
        # x_gconv=torch.einsum('bnki,nkio->bno',x_g,self.linear_weight)+self.linear_b
        x_gconv=self.linear(x_g)
        return x_gconv
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))
        # return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
class EmbGCN_SA(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(EmbGCN_SA, self).__init__()
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        # self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.sym_norm_Adj_matrix=F.softmax(torch.Tensor(adj).to(torch.device('cuda')))
        self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # x=self.SA(x)
        # node_num = node_embeddings.shape[0]
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # supports = torch.stack(support_set, dim=0)
        # 静态邻接矩阵
        # x_static = torch.einsum("nm,bmc->bmc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = self.linear(x_static)
        #
        x_sa = self.SA(x,self.sym_norm_Adj_matrix)
        x_sa=F.relu(x_sa)



        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out
        # supports=torch.einsum('knm,mm->knm',supports,self.sym_norm_Adj_matrix)
        #加空注

        # x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        #
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_sa
        # return x_gconv+(torch.tanh(x_gconv)+torch.sigmoid(x_static))
        # return x_gconv+torch.sigmoid(x_static)*x_static
        # return x_gconv+torch.sigmoid(x_static)*x_static+torch.sigmoid(x_gconv)*x_gconv
        # return torch.sigmoid(x_static)*x_static+(1-torch.sigmoid(x_static))*x_gconv
# class AVWGCN(nn.Module):
#     def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
#         super(AVWGCN, self).__init__()
#         self.cheb_k = cheb_k
#         self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
#         self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
#     def forward(self, x, node_embeddings):
#         #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
#         #output shape [B, N, C]
#
#         node_num = node_embeddings.shape[0]
#         supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
#         support_set = torch.eye(node_num).to(supports.device)+supports
#         #default cheb_k = 3
#         # for k in range(2, self.cheb_k):
#         #     print("cheb_kcheb_kcheb_kcheb_kcheb_k")
#         #     support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
#         # supports = torch.stack(support_set, dim=0)
#
#         weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
#         bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
#         x_g = torch.einsum("nm,bmi->bni", supports, x)      #B, cheb_k, N, dim_in
#         # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
#         x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
#         return x_gconv

class AVWGCN2(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(AVWGCN2, self).__init__()
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        # self.sym_norm_Adj_matrix = F.softmax(self.sym_norm_Adj_matrix, dim=1).to(device=torch.device("cuda"))
        # self.alpha = nn.Parameter(torch.FloatTensor([0.05]), requires_grad=True)  # D
        # self.beta = nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)  # S
        # self.Linear=nn.Linear(dim_in,dim_out,bias=True)
        # self.linear_weight=nn.Parameter(torch.FloatTensor(170,cheb_k,dim_in,dim_out),requires_grad=True)
        # self.linear_b=nn.Parameter(torch.FloatTensor(170,dim_out),requires_grad=True)
        # self.att_score=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
        self.cheb_k = cheb_k
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        # print(x.shape)
        # x,_=self.att_score(x)
        # x=self.Linear(x)
        # print(x.shape,"att_shape")
        # yuanlai
        node_num = node_embeddings.shape[0]
        # # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))+self.att_score(x)[0]), dim=1) # N N
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports = torch.eye(node_num).to(supports.device)+ supports
        # #
        # support_set = [torch.eye(node_num).to(supports.device), supports]
        # #default cheb_k = 3
        # for k in range(2, self.cheb_k):
        #     print("cheb_kcheb_kcheb_kcheb_kcheb_k")
        #     support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        # supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out

        # 修改
        # supports=supports+torch.eye(node_num).to(supports.device) # n n
        # x_g = torch.einsum("nn,bnc->bnc", supports, x)    #
        # x_gconv = self.Linear(x_g)

        # TA-GCN_SA
        # score,_=self.att_score(x) # b n n
        # score=torch.einsum('bnn,nn->bnn',score.to(torch.device('cuda')),self.sym_norm_Adj_matrix)
        # score=torch.einsum('bnm,bmc->bnc',score,x)
        # x_gconv=self.Linear(score)

        # score=nn.functional.relu(score)
        # att_out=torch.einsum('bnn,bnc->bnc')
        # att_out=torch.einsum()
        # print(self.sym_norm_Adj_matrix.shape,"aaaa")
        # score=torch.einsum("bnn,nn->bnn",score,self.sym_norm_Adj_matrix)
        # score=torch.matmul(score,self.sym_norm_Adj_matrix)
        # # print(score.shape)
        # # print(supports.shape)
        # # print(supports[0])
        # supports=torch.einsum("bnn,knm->bknm",score,supports)
        # # print(supports.shape)
        # x_g = torch.einsum("bknm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in


        # supports = self.alpha*score + self.beta*supports  # 加上空间注意力
        # supports=torch.einsum("nn,knm->knm",self.alpha*self.sym_norm_Adj_matrix,supports)# 加上静态邻接矩阵

        # static_out=torch.einsum("mm,bmc->bmc",self.sym_norm_Adj_matrix,x)
        # static_out=self.Linear(static_out) # b n o
        # static_out=nn.ReLU(static_out)

        # x_gconv=torch.einsum('bnki,nkio->bno',x_g,self.linear_weight)+self.linear_b # TA-GCN_linear
        # 不改
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out

        # no cheb_k
        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B,  N, dim_in
        # x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out

        # print(x_gconv.shape)
        # static_out=torch.einsum("nn,bnc->bnc",self.sym_norm_Adj_matrix,x)
        # print(static_out.shape)
        # gcn_out=self.alpha*score+self.beta*x_gconv
        # gcn_out=torch.relu(x_gconv)
        # gcn_out=x

        return x_gconv

if __name__=="__main__":
    x=torch.randn(64,170,1)
    adj=torch.randn(170,170)
    emb=torch.randn(170,2)
    gcn=EmbGCN(1,1,adj,2,2)
    out=gcn(x,emb)
    print(out.shape)
