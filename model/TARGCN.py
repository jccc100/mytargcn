import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.GRU import GRU
from model.temporal_attention_layer import TA_layer
from model.TCN import TemporalConvNet as tcn
from model.EmbGCN import EmbGCN
from torch.autograd import Variable
import math
device=torch.device('cuda')


class TARGCN_cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,num_layers=1,is_noTA=False):
        super(TARGCN_cell, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.is_noTA=is_noTA
        self.adj=adj
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))
        if is_noTA==False:
            self.TA = TA_layer(dim_out, dim_out, 2, 2)

    def forward(self, x, init_state, node_embeddings):

        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]

        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        if self.is_noTA==False:
            current_inputs=self.TA(current_inputs)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)
class TCN_cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,num_layers=1,is_noTA=False):
        super(TCN_cell, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.is_noTA=is_noTA
        self.adj=adj
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))
        if is_noTA==False:
            self.TA = TA_layer(dim_out, dim_out, 2, 2)

        self.gcns=nn.ModuleList()
        self.tcns=nn.ModuleList()
        self.gcn_num=3
        self.tcn_num=16
        for i in range(3):
            self.gcns.append(EmbGCN(dim_in,dim_out,adj,cheb_k,embed_dim))

        self.tcns.append(tcn(dim_in, [1, 1, 1], 3, 0.2))

    def forward(self, x, init_state, node_embeddings):

        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]

        current_inputs = x
        gcn_output=[]
        for i in range(self.gcn_num):
            for t in range(seq_length):
                x_gcn = self.gcns[i](x[:, t, :, :],node_embeddings)
                gcn_output.append(x_gcn)
            current_inputs=torch.stack(gcn_output,dim=1) # b t n d

        b, t, n, d = x.shape
        for j in range(self.tcn_num):
            current_inputs = current_inputs.permute(0, 2, 3, 1)  # b n d t
            current_inputs = current_inputs.reshape(b * n, d, t)  # b*n d t
            # tcn_out = self.tcn(x).reshape(b, n, d, t).permute(0, 3, 1, 2)  # [b*n d t] --> [b n d t] -->[b t n d]
            # current_inputs = x + tcn_out
            tcn_out=self.tcns[j](current_inputs).reshape(b, n, d, t).permute(0, 3, 1, 2) # b t n d



        # output_hidden = []
        # for i in range(self.num_layers):
        #     state = init_state[i]
        #     inner_states = []
        #     for t in range(seq_length):
        #         state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
        #         inner_states.append(state)
        #     output_hidden.append(state)
        #     current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        if self.is_noTA==False:
            current_inputs=self.TA(tcn_out)
        return current_inputs, 1

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)
class TARGCN(nn.Module):
    def __init__(self, args,adj=None):
        super(TARGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.adj=adj
        # self.default_graph = args.default_graph

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        # self.encoder = TARGCN_cell(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
        #                         args.embed_dim,self.adj, args.num_layers)
        self.encoder = TCN_cell(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                   args.embed_dim, self.adj, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(6, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -6:, :, :]                                   #B, 6, N, hidden
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # b t c n
        output = output.permute(0, 1, 3, 2)                             #B, T(12), N, C
        return output

