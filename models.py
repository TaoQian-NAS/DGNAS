import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, act, dropout):
        super(MLP, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, nhid))
        for _ in range(nlayer - 2):
            self.lins.append(nn.Linear(nhid, nhid))
        self.lins.append(nn.Linear(nhid, nclass))
        self.activation = act_map(act)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class LR(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class ResMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, act, dropout):
        super(ResMLP, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(nhid, nhid) for i in range(nlayer - 2)])
        self.fcs.insert(0, nn.Linear(nfeat, nhid))
        self.fcs.append(nn.Linear(nhid, nclass))
        self.activation = act_map(act)
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            temp = x.clone().detach()
            x = F.dropout(x, self.dropout, training=self.training)
            if i == 0:
                x = self.activation(self.fcs[i](x))
            else:
                x = self.activation(self.fcs[i](x)) + temp
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)


class InitialMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, act, dropout):
        super(InitialMLP, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(nhid, nhid) for i in range(nlayer - 2)])
        self.fcs.insert(0, nn.Linear(nfeat, nhid))
        self.fcs.append(nn.Linear(nhid, nclass))
        self.activation = act_map(act)
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            if i == 0:
                x = self.activation(self.fcs[i](x))
                initialx = x
            else:
                x = self.activation(self.fcs[i](x)) + initialx
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)


class DenseMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, act, dropout):
        super(DenseMLP, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(nhid, nhid) for i in range(nlayer - 2)])
        self.fcs.insert(0, nn.Linear(nfeat, nhid))
        self.fcs.append(nn.Linear(nhid, nclass))
        self.hidden = nhid
        self.activation = act_map(act)
        self.dropout = dropout

    def forward(self, x):
        temp = torch.zeros((len(self.fcs) - 1, *(*list(x.shape[:-1]), self.hidden)),
                           dtype=x.dtype, device=x.device)
        for i in range(len(self.fcs) - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation(self.fcs[i](x)) + torch.sum(temp, dim=0)
            x = x.div_(i + 1)
            temp[i] = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)


class JumpingMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, act, dropout):
        super(JumpingMLP, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(nhid, nhid) for i in range(nlayer - 2)])
        self.fcs.insert(0, nn.Linear(nfeat, nhid))
        self.fcs.append(nn.Linear(nhid * (nlayer - 1), nclass))
        self.hidden = nhid
        self.activation = act_map(act)
        self.dropout = dropout

    def forward(self, x):
        layer_outputs = []
        for i in range(len(self.fcs) - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation(self.fcs[i](x))
            layer_outputs.append(x)

        h = torch.cat(layer_outputs, dim=1)
        x = self.fcs[-1](h)
        return F.log_softmax(x, dim=1)


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


def initialModel(skip_connection, nfeat, nhid, nclass, nlayer, act, dropout):
    if nlayer == 1:
        return LR(nfeat=nfeat, nclass=nclass, dropout=dropout)
    elif skip_connection == "Residual":
        return ResMLP(nfeat=nfeat,
                      nhid=nhid,
                      nclass=nclass,
                      nlayer=nlayer,
                      act=act,
                      dropout=dropout)
    elif skip_connection == "Initial":
        return InitialMLP(nfeat=nfeat,
                          nhid=nhid,
                          nclass=nclass,
                          nlayer=nlayer,
                          act=act,
                          dropout=dropout)
    elif skip_connection == "Dense":
        return DenseMLP(nfeat=nfeat,
                        nhid=nhid,
                        nclass=nclass,
                        nlayer=nlayer,
                        act=act,
                        dropout=dropout)
    elif skip_connection == "Jumping":
        return JumpingMLP(nfeat=nfeat,
                          nhid=nhid,
                          nclass=nclass,
                          nlayer=nlayer,
                          act=act,
                          dropout=dropout)
    else:
        return MLP(nfeat=nfeat,
                   nhid=nhid,
                   nclass=nclass,
                   nlayer=nlayer,
                   act=act,
                   dropout=dropout)


def aggregator(aggregator_type, target):
    if aggregator_type == "max":
        return torch.max(target, dim=0).values
    elif aggregator_type == "min":
        return torch.min(target, dim=0).values
    elif aggregator_type == "mean":
        return torch.mean(target, dim=0)
    elif aggregator_type == "sum":
        return torch.sum(target, dim=0)
    else:
        raise Exception("wrong aggregator type")
