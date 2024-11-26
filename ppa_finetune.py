import os
os.system('pip install torch_geometric')
from shutil import copyfile

for lib in ['utilz', 'ppa_models', 'ppa_op_graph_classification', 'ppa_operations', 'ppa_utils', 'ppa_genotypes', 'ppa_message_passing', 'ppa_pyg_gnn_layer', 'ppa_agg_zoo', 'ppa_inits', 'deepergcn_with_hig', 'ppa_conv', 'ppa_algos', 'ppa_operations']:
    copyfile(src = f'/kaggle/usr/lib/{lib}/{lib}.py', dst = f'/kaggle/working/{lib}.py')
from utilz import flag, warm_up_lr
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np
from torch_geometric.loader import DataLoader

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import random

import torch
import statistics
from tqdm import tqdm

from ppa_models import NetworkGNN as Network

import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np
from torch_geometric.loader import DataLoader

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
import random

import torch
import statistics
from tqdm import tqdm

### from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ppa_models import NetworkGNN as Network
def save_ckpt(model, optimizer, loss, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        # print("Directory ", save_path, " is created.")

    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    # print('model has been saved as {}'.format(filename))
graph_classification_dataset=['DD', 'MUTAG', 'PROTEINS', 'NCI1', 'NCI109','IMDB-BINARY', 'REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI','COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa']
node_classification_dataset = ['Cora', 'CiteSeer', 'PubMed', 'Amazon_Computers', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Photo']

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 0)')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--loss', type=str, default='auroc', help='')
    parser.add_argument('--data', type=str, default='ogbg-ppa', help='location of the data corpus')
    parser.add_argument('--model_save_path', type=str, default='model_finetune',
                        help='the directory used to save models')
    parser.add_argument('--add_virtual_node', action='store_true')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=5, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--use_hyperopt', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--with_layernorm_learnable', action='store_true', default=False, help='use the learnable layer norm')
    parser.add_argument('--BN',  action='store_true', default=True,  help='use BN.')
    # parser.add_argument('--flag', action='store_true', default=True,  help='use flag.')
    parser.add_argument('--flag', type=str, default='true')

    parser.add_argument('--feature', type=str, default='full',
                        help='two options: full or simple')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--optimizer', type=str, default='pesg', help='')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate set for optimizer.')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of data.')
    parser.add_argument('--model', type=str, default='SANE')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--is_mlp', action='store_true', default=False, help='is_mlp')
    parser.add_argument('--ft_weight_decay', action='store_true', default=False, help='with weight decay in finetune stage.')
    parser.add_argument('--ft_dropout', action='store_true', default=False, help='with dropout in finetune stage')
    parser.add_argument('--ft_mode', type=str, default='811', choices=['811', '622', '10fold'], help='data split function.')
    parser.add_argument('--hyper_epoch', type=int, default=1, help='hyper epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs for each model')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warm_up epochs for each model')
    parser.add_argument('--cos_lr',  action='store_true', default=True,  help='use cos lr.')
    parser.add_argument('--lr_min',  type=float, default=0.005,  help='use cos lr.')
    parser.add_argument('--show_info',  action='store_true', default=True,  help='print training info in each epoch')
    parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
    parser.add_argument('--search_act', action='store_true', default=False, help='search act in supernet.')
    parser.add_argument('--one_pooling', action='store_true', default=False, help='only one pooling layers after 2th layer.')
    parser.add_argument('--seed', type=int, default=777, help='seed for finetune')
    parser.add_argument('--remove_pooling', action='store_true', default=True,
                        help='remove pooling block.')
    parser.add_argument('--remove_readout', action='store_true', default=True,
                        help='remove readout block. Only search the last readout block.')
    parser.add_argument('--remove_jk', action='store_true', default=False,
                        help='remove ensemble block. In the last readout block,use global sum. Graph representation = Z3')
    parser.add_argument('--fixpooling', type=str, default='null',
                        help='use fixed pooling functions')
    parser.add_argument('--fixjk',action='store_true', default=False,
                        help='use concat,rather than search from 3 ops.')

    # flag
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--save', type=str, default='EXP', help='experiment nam ')

    global args
    args = parser.parse_args()
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args.seed))

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, device, loader, optimizer, multicls_criterion, grad_clip=0.):
    loss_list = []
    model.train()
    iters = len(loader)

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1))

            loss.backward()
            loss_list.append(loss.item())

            if grad_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    grad_clip)

            optimizer.step()
            if args.cos_lr:
                pass
                    #cos_lr_warmrestarts
                    # scheduler.step(args.epochs + step / iters)


    return statistics.mean(loss_list)

def train_with_flag(model, device, loader, optimizer, multicls_criterion):
    loss_list = []
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            forward = lambda perturb: model(batch, perturb).to(torch.float32)
            model_forward = (model, forward)
            y = batch.y.view(-1, )
            perturb_shape = (batch.x.shape[0], args.hidden_size)
            loss, _ = flag(model_forward, perturb_shape, y, optimizer, device, multicls_criterion)
            loss_list.append(loss.item())

    # print(total_loss/len(loader))
    return statistics.mean(loss_list)


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            # pred = torch.sigmoid(pred)
            y_true.append(batch.y.view(-1, 1).detach().cpu()) # remove random forest pred
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())


    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    return evaluator.eval(input_dict)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

