# %% [code]
# %% [code]

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shutil import copyfile
os.system('pip install cython')

import pyximport
import sys
sys.path.append('/kaggle/working/')
pyximport.install(setup_args={'include_dirs': np.get_include()})
copyfile(src = '/kaggle/input/sososo/algos.cpython-310-x86_64-linux-gnu.so', dst = '/kaggle/working/algos.so')
import algos

copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_inits.py', dst = '/kaggle/working/ppa_inits.py')


os.system('pip install torch_geometric')
from torch_geometric.utils import add_self_loops,remove_self_loops
from torch_geometric.nn import  global_mean_pool,global_add_pool

# from operations import *
copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_op_graph_classification.py', dst = '/kaggle/working/ppa_op_graph_classification.py')
from ppa_op_graph_classification import *

copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_genotypes.py', dst = '/kaggle/working/ppa_genotypes.py')
from ppa_genotypes import NA_PRIMITIVES, NA_PRIMITIVES2, SC_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES, SE_PRIMITIVES, FU_PRIMITIVES
from ppa_genotypes import Genotype

copyfile(src = '/kaggle/usr/lib/ppa_inits/ppa_inits.py', dst = '/kaggle/working/ppa_inits.py')
copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_genotypes.py', dst = '/kaggle/working/ppa_genotypes.py')

copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_pooling_zoo.py', dst = '/kaggle/working/ppa_pooling_zoo.py')
from ppa_pooling_zoo import filter_features,filter_perm

copyfile(src = '/kaggle/usr/lib/ppa_op_graph_classification/ppa_pooling_zoo.py', dst = '/kaggle/working/ppa_pooling_zoo.py')




class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        #assert self.mode in ['cat']

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)
        if self.mode != 'last':
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
    
def act_map(act):
    # if act == "linear":
    #     return lambda x: x
    if act == "elu":
        return torch.nn.ELU
    elif act == "sigmoid":
        return torch.nn.Sigmoid
    elif act == "tanh":
        return torch.nn.Tanh
    elif act == "relu":
        return torch.nn.ReLU
    elif act == "relu6":
        return torch.nn.ReLU6
    elif act == "softplus":
        return torch.nn.Softplus
    elif act == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise Exception("wrong activate function")
#
# def act_map(act):
#     if act == "linear":
#         return lambda x: x
#     elif act == "elu":
#         return torch.nn.functional.elu()
#     elif act == "sigmoid":
#         return torch.sigmoid()
#     elif act == "tanh":
#         return torch.tanh()
#     elif act == "relu":
#         return torch.nn.functional.relu()
#     elif act == "relu6":
#         return torch.nn.functional.relu6()
#     elif act == "softplus":
#         return torch.nn.functional.softplus()
#     elif act == "leaky_relu":
#         return torch.nn.functional.leaky_relu()
#     else:
#         raise Exception("wrong activate function")
class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, with_linear):
    super(NaMixedOp, self).__init__()
    self._ops = nn.ModuleList()

    for primitive in NA_PRIMITIVES:
      op = NA_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

      if with_linear:
        self._ops_linear = nn.ModuleList()
        op_linear = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear.append(op_linear)

      # self.act = act_map(act)

  def forward(self, x, weights, edge_index, edge_weights, with_linear, edge_attr):
    mixed_res = []
    if with_linear:
      for w, op, linear in zip(weights, self._ops, self._ops_linear):
        mixed_res.append(w * (op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr)+linear(x)))
        # print('with linear')
    else:
      for w, op in zip(weights, self._ops):
        mixed_res.append(w * (op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr)))
        # print('without linear')
    return sum(mixed_res)

class SeMixedOp(nn.Module):

  def __init__(self):
    super(SeMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SE_PRIMITIVES:
        #OP只有两种 True 和 False
      op = SE_OPS[primitive]()
      self._ops.append(op)

  def forward(self, h_list, weights):
    index = len(h_list)
    mixed_res = []
    #weights 第i层有 2*（i-1）个
    # print(weights)

    for i in range(index):
        tmp = []
        for w, op in zip(weights[i], self._ops):
            tmp.append(w * op(h_list[i]))
            # print(tmp)

        mixed_res.append(sum(tmp))

    return mixed_res

class FuMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(FuMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in FU_PRIMITIVES:
      op = FU_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x_list, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * op(x_list))
    return sum(mixed_res)

class ScMixedOp(nn.Module):

  def __init__(self):
    super(ScMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SC_PRIMITIVES:
      op = SC_OPS[primitive]()
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * op(x))
    return sum(mixed_res)

class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      # mixed_res.append(w * F.relu(op(x)))
      mixed_res.append(w * F.elu(op(x)))
    return sum(mixed_res)
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.float64, device=index.device)
    new_index = index.fill_(index[0]).type(torch.long)
    mask[new_index] = 1.0
    return mask
class PoolingMixedOp(nn.Module):
    def __init__(self, hidden, ratio, num_nodes=0):
        super(PoolingMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in POOL_PRIMITIVES:
            op = POOL_OPS[primitive](hidden, ratio, num_nodes)
            self._ops.append(op)

    def forward(self, x, edge_index, edge_weights, data, batch, mask, weights):
        new_x = []
        new_edge_weight = []
        new_perm = []
        # neither add or ewmove self_loop, so edge_index remain unchanged.
        for w, op in zip(weights, self._ops):
            # mixed_res.append(w * F.relu(op(x)))
            x_tmp, edge_index, edge_weight_tmp, batch, perm = op(x, edge_index, edge_weights, data, batch, mask)
            #print(perm.size(),w)
            new_x.append(x_tmp * w)
            new_edge_weight.append(w * edge_weight_tmp)
            new_perm.append(w * index_to_mask(perm, x.size(0)))
        return sum(new_x), edge_index, sum(new_edge_weight), batch, perm

        # #remove the nodes
        # x, edge_index, edge_weight, batch, perm = filter_features(sum(new_x), edge_index, sum(new_edge_weight), batch, th=0.001)
        # return x, edge_index, edge_weight, batch, perm

        #remove nodes with perm
        #x, edge_index, edge_weight, batch, perm = filter_perm(sum(new_x), edge_index, sum(new_edge_weight), batch, sum(new_perm), th=0.01)
        #return x, edge_index, edge_weight, batch, perm

        # return x, edge_index, edge_weight, batch, perm
        # for w, op in zip(weights, self._ops):
        #     if w == 0:
        #         continue
        #     else:
        #         new_x, new_edge_index, new_batch, new_mask = op(x, edge_index, data, batch, mask)
        # return new_x, new_edge_index, new_batch, new_mask

class ReadoutMixedOp(nn.Module):
    def __init__(self, hidden):
        super(ReadoutMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in READOUT_PRIMITIVES:
            op = READOUT_OPS[primitive](hidden)
            self._ops.append(op)

    def forward(self, x, batch, mask, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            tmp_res = w * op(x, batch, mask)
            # print('readout', tmp_res.size())
            mixed_res.append(tmp_res)
        return sum(mixed_res)

class ActMixedOp(nn.Module):
    def __init__(self):
        super(ActMixedOp, self).__init__()
        self._ops = nn.ModuleDict()
        for primitive in ACT_PRIMITIVES:
            if primitive == 'linear':
                self._ops[primitive] = act_map(primitive)
            else:
                self._ops[primitive] = act_map(primitive)()

    def forward(self, x,  weights):
        mixed_res = []

        for i in range(len(ACT_PRIMITIVES)):
            mixed_res.append(weights[i] * self._ops[ACT_PRIMITIVES[i]](x))
        return sum(mixed_res)

class Network(nn.Module):
  '''
      implement this for sane.
      Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
      for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
  '''

  def __init__(self, dataset, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, args=None, with_conv_linear=False,num_nodes=0 ):
    super(Network, self).__init__()

    self.dataset = dataset
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_nodes = num_nodes
    self._criterion = criterion
    self.dropout = dropout
    self.epsilon = epsilon
    self.with_linear = with_conv_linear
    self.explore_num = 0
    self.args = args
    self.temp = args.temp
    self._loc_mean = args.loc_mean
    self._loc_std = args.loc_std

    self.pool = global_mean_pool

    # In adaptive stage.a
    self.delta = self.args.delta
    self.N = len(NA_PRIMITIVES)*3 + len(SC_PRIMITIVES)*2 +len(LA_PRIMITIVES)
    self.s = np.zeros(self.N)
    self.Delta = 1.
    self.gamma = 0.
    self.alpha = 1.5
    self.Delta_max = 10.
    # if num_layers == 1:
    #     self.pooling_ratio = [0.1]
    # elif num_layers == 2:
    #     self.pooling_ratio = [0.25, 0.25]
    # elif num_layers == 3:
    #     self.pooling_ratio = [0.5, 0.5, 0.5]
    # elif num_layers == 4:
    #     self.pooling_ratio = [0.6, 0.6, 0.6, 0.6]
    # elif num_layers == 5:
    #     self.pooling_ratio = [0.7, 0.7, 0.7, 0.7, 0.7]
    # elif num_layers == 6:
    #     self.pooling_ratio = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    # elif num_layers == 7:
    #     self.pooling_ratio = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    # elif num_layers == 8:
    #     self.pooling_ratio = [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8]
    # elif num_layers == 9:
    #     self.pooling_ratio = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
    # elif num_layers == 10:
    #     self.pooling_ratio = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]
    # elif num_layers == 11:
    #     self.pooling_ratio = [1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11]
    # elif num_layers == 12:
    #     self.pooling_ratio = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12]
    # elif num_layers == 13:
    #     self.pooling_ratio = [1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13]
    # elif num_layers == 14:
    #     self.pooling_ratio = [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13,
    #                           1 / 13, 1 / 13, 1 / 13]

    # self.atom_encoder = AtomEncoder(hidden_size)
    self.node_encoder = torch.nn.Embedding(1, self.hidden_size)
    # self.init_mlp = torch.nn.Linear(self.hidden_size, self.hidden_size)
    # self.last_mlp = torch.nn.Linear(self.hidden_size, self.hidden_size)

    # node aggregator op
    self.gnn_layers = nn.ModuleList()
    # self.BNs = torch.nn.ModuleList()
    self.Se_layers = nn.ModuleList()
    self.Fu_layers = nn.ModuleList()
    for i in range(num_layers):
        self.gnn_layers.append(NaMixedOp(hidden_size, hidden_size, self.with_linear))
        # self.BNs.append(torch.nn.BatchNorm1d(self.hidden_size))
    for i in range(num_layers + 1):
        self.Se_layers.append(SeMixedOp())
        self.Fu_layers.append(FuMixedOp(self.hidden_size, i + 1))

    self.batch_norms = nn.ModuleList()
    for layer in range(self.num_layers + 1):
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))
        
    #act op

    #readoutop


    #pooling ops
    self.pool = global_add_pool

    #graph representation aggregator op
    # self.layer6 = LaMixedOp(hidden_size, num_layers+1)

    self.classifier = nn.Linear(hidden_size, out_dim)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._criterion, self.in_dim, self.out_dim, self.hidden_size).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def _get_categ_mask(self, alpha):
      # log_alpha = torch.log(alpha)
      log_alpha = alpha
      u = torch.zeros_like(log_alpha).uniform_()
      softmax = torch.nn.Softmax(-1)
      one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)
      return one_hot
  def asng_sampling(self,alpha):

      rand = torch.rand(1, alpha.size()[0])  # range of random number is [0, 1)
      rands = torch.zeros_like(alpha)
      for i in range(alpha.size()[1]):
          rands[:, i] = rand
      cum_theta = alpha.cumsum(axis=1)  # the same shape as
      x = (cum_theta - alpha <= rands) & (rands < cum_theta)
      return x.float()

  def get_one_hot_alpha(self, alpha):
      one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
      idx = torch.argmax(alpha, dim=-1)

      for i in range(one_hot_alpha.size(0)):
        one_hot_alpha[i, idx[i]] = 1.0

      return one_hot_alpha

  # other feature computation
  def convert_to_single_emb(self, x, offset=512):
      feature_num = x.size(1) if len(x.size()) > 1 else 1
      feature_offset = 1 + \
                       torch.arange(0, feature_num * offset, offset, dtype=torch.long)
      x = x + feature_offset.to(x.device)
      return x

  def forward(self, data, discrete=False, mode='none'):

    # search act
    # self.args.search_act = False
    # data = self.preprocess_item(data)
    with_linear = self.with_linear
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    print(edge_attr.shape)
    batch = data.batch

    if self.args.model_type == 'darts':
        na_alphas = F.softmax(self.log_na_alphas / 1e-3, dim=-1)
        fu_alphas = F.softmax(self.log_fu_alphas / 1e-3, dim=-1)
        se_alphas = []
        for alpha in self.log_se_alphas:
            se_alphas.append(F.softmax(alpha / 1e-3, dim=-1))
        # se_alphas = F.softmax(self.log_se_alphas / 1e-3, dim=-1)
        la_alphas = F.softmax(self.log_la_alphas / 1e-3, dim=-1)
        readout_alphas = F.softmax(self.log_readout_alphas / 1e-3, dim=-1)
        act_alphas = F.softmax(self.log_act_alphas / 1e-3, dim=-1)
        print('DARTS: sampled arch in train w', self._sparse(se_alphas, fu_alphas, na_alphas, act_alphas, readout_alphas, la_alphas))
    else:

        na_alphas = self._get_categ_mask(self.log_na_alphas)
        # sc_alphas = self._get_categ_mask(self.log_sc_alphas)
        la_alphas = self._get_categ_mask(self.log_la_alphas) #eg. la_alphas: tensor([[0., 1., 0.]]
        pool_alphas = self._get_categ_mask(self.log_pool_alphas)
        readout_alphas = self._get_categ_mask(self.log_readout_alphas)
        act_alphas = self._get_categ_mask(self.log_act_alphas)
        # print('alpha in train w:',self._arch_parameters)
        # print('sampled arch in train w', self._sparse(na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas))
    if mode == 'evaluate_single_path':
        na_alphas = self.get_one_hot_alpha(na_alphas)
        la_alphas = self.get_one_hot_alpha(la_alphas)
        pool_alphas = self.get_one_hot_alpha(pool_alphas)
        readout_alphas = self.get_one_hot_alpha(readout_alphas)
        act_alphas = self.get_one_hot_alpha(act_alphas)

    graph_representations = []
    # x = self.atom_encoder(x)
    x = self.node_encoder(x)
    xs = []
    edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()

    # if self.args.remove_readout:
    #     graph_representations.append(x)
    #     # graph_representations.append(global_add_pool(x, batch))
    # else:
    #     graph_representations.append(self.readout_layers[0](x, batch, None, readout_alphas[0]))


    # x = self.gnn_layers[0](x, na_alphas[0], edge_index, edge_weights, with_linear, edge_attr)
    # xs.append(x)
    h_list = []

    # x = F.elu(self.init_mlp(x))
    # x = F.dropout(x, p=self.dropout, training=self.training)
    h_list.append(x)

    for i in range(0, self.num_layers):


        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # graph_representations[i] += virtualnode_embedding[batch]
        x_list = self.Se_layers[i](h_list, se_alphas[i])
        x2 = self.Fu_layers[i](x_list, fu_alphas[i])
        x2 = self.batch_norms[i](x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.gnn_layers[i](x2, na_alphas[i], edge_index, edge_weights, with_linear, edge_attr)
        # x = F.elu(x)
        # x = F.elu(x)

        # dropout
        # x = F.dropout(x, p=self.dropout, training=self.training)

        h_list.append(x)

    x_list = self.Se_layers[i + 1](h_list, se_alphas[i + 1])
    x2 = self.Fu_layers[i + 1](x_list, fu_alphas[i + 1])

    x2 = self.batch_norms[i + 1](x2)
    x2 = F.elu(x2)
    x2 = F.dropout(x2, p=self.dropout, training=self.training)
    # x = F.elu(self.last_mlp(x2))


    x = self.pool(x2, batch)

    x = self.classifier(x)

    return x, [se_alphas, fu_alphas, na_alphas, act_alphas, readout_alphas, la_alphas]

    # return F.log_softmax(logits, dim=-1), [na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas]
    # return logits, [na_alphas, act_alphas, pool_alphas, readout_alphas, la_alphas]

  def _loss(self, data, is_valid=True):
      input = self(data).cuda()
      target = data.y.cuda()
      return self._criterion(input, target)


  def _initialize_alphas(self):
    #k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_na_ops = len(NA_PRIMITIVES)
    num_se_ops = len(SE_PRIMITIVES)
    num_fu_ops = len(FU_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)
    num_pool_ops = len(POOL_PRIMITIVES)
    num_readout_ops = len(READOUT_PRIMITIVES)
    num_act_ops = len(ACT_PRIMITIVES)

    self.log_se_alphas = []
    if self.args.model_type == 'darts':
        for i in range(self.num_layers + 1):
            self.log_se_alphas.append(Variable(1e-3 * torch.randn(i + 1, num_se_ops).cuda(), requires_grad=True))

        self.log_fu_alphas = Variable(1e-3 * torch.randn(self.num_layers + 1, num_fu_ops).cuda(), requires_grad=True)
        self.log_na_alphas = Variable(1e-3*torch.randn(self.num_layers, num_na_ops).cuda(), requires_grad=True)
        self.log_act_alphas = Variable(1e-3*torch.randn(self.num_layers, num_act_ops).cuda(), requires_grad=True)
        self.log_pool_alphas = Variable(1e-3*torch.randn(self.num_layers, num_pool_ops).cuda(), requires_grad=True)
        self.log_readout_alphas = Variable(1e-3*torch.randn(self.num_layers+1, num_readout_ops).cuda(), requires_grad=True)
        self.log_la_alphas = Variable(1e-3*torch.randn(1, num_la_ops).cuda(), requires_grad=True)


    else:
        self.log_na_alphas = Variable(
            torch.ones(self.num_layers, num_na_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
        self.log_act_alphas = Variable(
            torch.ones(self.num_layers, num_act_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)

        self.log_pool_alphas = Variable(
            torch.ones(self.num_layers, num_pool_ops).normal_(self._loc_mean, self._loc_std).cuda(), requires_grad=True)
        self.log_readout_alphas = Variable(
            torch.ones(self.num_layers + 1, num_readout_ops).normal_(self._loc_mean, self._loc_std).cuda(),
            requires_grad=True)

        self.log_la_alphas = Variable(torch.ones(1, num_la_ops).normal_(self._loc_mean, self._loc_std).cuda(),
                                      requires_grad=True)


    self._arch_parameters = [
      # self.log_se_alphas,
      self.log_fu_alphas,
      self.log_na_alphas,
      self.log_act_alphas,
      self.log_readout_alphas,
      self.log_la_alphas
    ]
    for log_se_alpha in self.log_se_alphas:
        self._arch_parameters.append(log_se_alpha)

  def arch_parameters(self):
    return self._arch_parameters

  def _sparse(self, se_alphas, fu_alphas, na_weights, act_alphas, readout_alphas, la_weights):
      gene = []

      for se_alpha in se_alphas:
          se_indices = torch.argmax(se_alpha, dim=-1)
          for k in se_indices:
              gene.append(SE_PRIMITIVES[k])

      fu_indices = torch.argmax(fu_alphas, dim=-1)
      for k in fu_indices:
          gene.append(FU_PRIMITIVES[k])
      # sc_indices = sc_weights.argmax(dim=-1)

      na_indices = torch.argmax(na_weights, dim=-1)
      for k in na_indices:
          gene.append(NA_PRIMITIVES[k])
      #sc_indices = sc_weights.argmax(dim=-1)

      act_indices = torch.argmax(act_alphas,dim=-1)
      for k in act_indices:
          gene.append(ACT_PRIMITIVES[k])


      readout_indices = torch.argmax(readout_alphas,dim=-1)
      for k in readout_indices:
          gene.append(READOUT_PRIMITIVES[k])

      la_indices = torch.argmax(la_weights, dim=-1)
      for k in la_indices:
          gene.append(LA_PRIMITIVES[k])
      return '||'.join(gene)
  def genotype(self):

    #gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    list_log_se_alphas = []
    for log_se_alpha in self.log_se_alphas:
        list_log_se_alphas.append(F.softmax(log_se_alpha, dim=-1).data.cpu())


    gene = self._sparse(
                        list_log_se_alphas,
                        F.softmax(self.log_fu_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_na_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_act_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_readout_alphas, dim=-1).data.cpu(),
                        F.softmax(self.log_la_alphas, dim=-1).data.cpu())
    return gene
# def get_arch(self):
  def sample_arch(self):

    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)

    gene = []
    for i in range(3):
        op = np.random.choice(NA_PRIMITIVES, 1)[0]
        gene.append(op)
    for i in range(2):
        op = np.random.choice(SC_PRIMITIVES, 1)[0]
        gene.append(op)
    op = np.random.choice(LA_PRIMITIVES, 1)[0]
    gene.append(op)
    return '||'.join(gene)

  def get_weights_from_arch(self, arch):
    arch_ops = arch.split('||')
    #print('arch=%s' % arch)
    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)


    na_alphas = Variable(torch.zeros(3, num_na_ops).cuda(), requires_grad=True)
    sc_alphas = Variable(torch.zeros(2, num_sc_ops).cuda(), requires_grad=True)
    la_alphas = Variable(torch.zeros(1, num_la_ops).cuda(), requires_grad=True)

    for i in range(3):
        ind = NA_PRIMITIVES.index(arch_ops[i])
        na_alphas[i][ind] = 1

    for i in range(3, 5):
        ind = SC_PRIMITIVES.index(arch_ops[i])
        sc_alphas[i-3][ind] = 1

    ind = LA_PRIMITIVES.index(arch_ops[5])
    la_alphas[0][ind] = 1

    arch_parameters = [na_alphas, sc_alphas, la_alphas]
    return arch_parameters

  def set_model_weights(self, weights):
    self.na_weights = weights[0]
    self.sc_weights = weights[1]
    self.la_weights = weights[2]
    #self._arch_parameters = [self.na_alphas, self.sc_alphas, self.la_alphas]
