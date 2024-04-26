import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from args import initialize
from norms import Gnorm

from utils import load_data, accuracy, normalize_adj
from models import LR, MLP, ResMLP, DenseMLP, InitialMLP, initialModel, aggregator


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:03d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val, acc_test


args = initialize()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, _ = load_data(dataset=args.dataset)
n_nodes, feat_dim = features.shape
adj_norm = normalize_adj(adj)
labels = torch.LongTensor(labels)
t_total = time.time()
# for _ in range(args.hops):
#     features = torch.spmm(adj_norm, features)
#
x = features
y = []
y.append(features)
gnorm = Gnorm(args.graph_normalization)

for i in range(args.propagation_layers):
    x = torch.spmm(adj_norm, x).detach_()
    # x = torch.load("./propagation/ogbn-arxiv/hop" + str(i+1) + ".pth")
    x = gnorm.norm(x)
    # print(y.add_(x))
    y.append(x)

target = torch.stack(y, dim=0)
features = aggregator(args.aggregator_type, target).detach_()
# model = LR(nfeat=features.shape[1],
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)
model = initialModel(skip_connection=args.skip_connections, nfeat=features.shape[1],
                     nhid=args.dim_hidden,
                     nclass=labels.max().item() + 1,
                     nlayer=args.num_layers,
                     act=args.activation_function,
                     dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

model = model.to(device)
features = features.to(device)
labels = labels.to(device)

best_val = 0
best_test = 0
for epoch in range(args.epochs):
    acc_val, acc_test = train(epoch)
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
