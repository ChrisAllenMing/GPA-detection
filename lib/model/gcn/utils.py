import numpy as np
import scipy.sparse as sp
import torch
import time


def get_adj(rois, epsilon = 1e-6):
    # compute the area of every bbox
    area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
    area = area + (area == 0).float() * epsilon

    # compute iou
    x_min = rois[:,1]
    x_min_copy = torch.stack([x_min] * rois.size(0), dim=0)
    x_min_copy_ = x_min_copy.permute((1,0))
    x_min_matrix = torch.max(torch.stack([x_min_copy, x_min_copy_], dim=-1), dim=-1)[0]
    x_max = rois[:,3]
    x_max_copy = torch.stack([x_max] * rois.size(0), dim=0)
    x_max_copy_ = x_max_copy.permute((1, 0))
    x_max_matrix = torch.min(torch.stack([x_max_copy, x_max_copy_], dim=-1), dim=-1)[0]
    y_min = rois[:,2]
    y_min_copy = torch.stack([y_min] * rois.size(0), dim=0)
    y_min_copy_ = y_min_copy.permute((1, 0))
    y_min_matrix = torch.max(torch.stack([y_min_copy, y_min_copy_], dim=-1), dim=-1)[0]
    y_max = rois[:,4]
    y_max_copy = torch.stack([y_max] * rois.size(0), dim=0)
    y_max_copy_ = y_max_copy.permute((1, 0))
    y_max_matrix = torch.min(torch.stack([y_max_copy, y_max_copy_], dim=-1), dim=-1)[0]
    
    w = torch.max(torch.stack([(x_max_matrix - x_min_matrix), torch.zeros_like(x_min_matrix)], dim = -1), dim = -1)[0]
    h = torch.max(torch.stack([(y_max_matrix - y_min_matrix), torch.zeros_like(y_min_matrix)], dim = -1), dim = -1)[0]
    intersection = w * h
    area_copy = torch.stack([area] * rois.size(0), dim = 0)
    area_copy_ = area_copy.permute((1,0))
    area_sum = area_copy + area_copy_
    union = area_sum - intersection
    iou = intersection / union

    return iou

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    np_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    indices = torch.from_numpy(np_indices)
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
