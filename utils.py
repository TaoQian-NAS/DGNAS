import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl

import random
import networkx as nx

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from texttable import Texttable
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Coauthor, Amazon


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_planetoid_data(dataset, supervision=True):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("./data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "./data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if not supervision:
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)
    else:
        num_nodes = features.size(0)
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2
        idx_train = range(int(train_rate * num_nodes))
        idx_val = range(int(train_rate * num_nodes) + 1, int((train_rate + val_rate) * num_nodes))
        idx_test = range(int((train_rate + val_rate) * num_nodes) + 1, num_nodes)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, np.argmax(labels, 1), idx_train, idx_val, idx_test, nx.from_dict_of_lists(graph)


def load_ogbn_data(dataset_name):
    dataset = PygNodePropPredDataset(name=dataset_name, root='./data/', transform=T.ToUndirected())
    data = dataset[0]
    x = data.x
    y = data.y.squeeze()
    edge_index = data.edge_index
    adj = to_scipy_sparse_matrix(edge_index)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    print(y[:10])

    return adj, x, y, train_idx, valid_idx, test_idx, nx.from_scipy_sparse_matrix(adj)


def load_coathor_amazon_data(dataset_name):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    if dataset_name == 'cs' or dataset_name == 'physics':
        dataset = Coauthor('./data/', dataset_name, transform=T.ToUndirected())
    elif dataset_name == 'computers' or dataset_name == 'photo':
        dataset = Amazon('./data/', dataset_name, transform=T.ToUndirected())
    else:
        ValueError('wrong dataset')

    data = dataset[0]
    num_classes = dataset.num_classes
    adj = to_scipy_sparse_matrix(data.edge_index)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    test_index = torch.cat([i[50:] for i in indices], dim=0)
    test_index = test_index[torch.randperm(test_index.size(0))]

    return adj, data.x, data.y, train_index, val_index, test_index, nx.from_scipy_sparse_matrix(adj)


def load_data(dataset):
    planetoid_name = ['cora', 'citeseer', 'pubmed']
    ogbn_name = ['ogbn-arxiv','ogbn-products']
    coautor_amazon_name = ['cs', 'physics', 'computers', 'photo']

    if dataset in planetoid_name:
        return load_planetoid_data(dataset)
    elif dataset in ogbn_name:
        return load_ogbn_data(dataset)
    elif dataset in coautor_amazon_name:
        return load_coathor_amazon_data(dataset)
    else:
        raise ValueError('Wrong dataset name')


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx, r=0.5):
    """Row-normalize sparse matrix"""
    mx = sp.coo_matrix(mx) + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt_left = np.power(rowsum, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(rowsum, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_normalized = mx.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def print_args(args):
    _dict = vars(args)
    _key = sorted(_dict.items(), key=lambda x: x[0])
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k, _ in _key:
        t.add_row([k, _dict[k]])
    print(t.draw())


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range
