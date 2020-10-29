import numpy as np
import os
import copy
import collections
import scipy.io as sio
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import sys
import itertools


Datasets = collections.namedtuple('Datasets', ['train', 'test', 'embeddings', 'node_cluster',
                                                'labels', 'idx_label', 'label_name'])

class DataSet(object):

    def __init__(self, edge, nums_type, **kwargs):
        self.edge = edge
        self.edge_set = set(map(tuple, edge)) ### ugly code, need to be fixed
        self.nums_type = nums_type
        self.kwargs = kwargs
        self.nums_examples = len(edge)
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, embeddings, batch_size=16, num_neg_samples=1, pair_radio=0.9, sparse_input=True):
        """
            Return the next `batch_size` examples from this data set.
            if num_neg_samples = 0, there is no negative sampling.
        """
        while 1:
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.nums_examples:
                self.epochs_completed += 1
                np.random.shuffle(self.edge)
                start = 0
                self.index_in_epoch = batch_size
                assert self.index_in_epoch <= self.nums_examples
            end = self.index_in_epoch
            neg_data = []
            # 采样
            for i in range(start, end): # 每个正样本对一个num_neg_samples 个负样本
                n_neg = 0
                while(n_neg < num_neg_samples): # 制作负样本，
                    ### warning !!! we need deepcopy to copy list
                    index = copy.deepcopy(self.edge[i])
                    mode = np.random.rand()
                    if mode < pair_radio: # 有成对信息的负样本
                        type_ = np.random.randint(3)
                        node = np.random.randint(self.nums_type[type_])
                        index[type_] = node
                    else: # 全随机的负样本
                        types_ = np.random.choice(3, 2, replace=False)
                        node_1 = np.random.randint(self.nums_type[types_[0]])
                        node_2 = np.random.randint(self.nums_type[types_[1]])
                        index[types_[0]] = node_1
                        index[types_[1]] = node_2
                    if tuple(index) in self.edge_set:
                        continue
                    n_neg += 1
                    neg_data.append(index)
            if len(neg_data) > 0:
                batch_data = np.vstack((self.edge[start:end], neg_data))#[16:15*5]=positive:negative
                nums_batch = len(batch_data)
                labels = np.zeros(nums_batch)
                labels[0:end-start] = 1 # 正样本 标签为1
                perm = np.random.permutation(nums_batch)
                batch_data = batch_data[perm] # 打乱顺序
                labels = labels[perm]
            else:
                batch_data = self.edge[start:end]
                nums_batch = len(batch_data)
                labels = np.ones(len(batch_data))
            batch_e = embedding_lookup(embeddings, batch_data, sparse_input)
            yield (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                    dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)]+[('classify_layer', labels)]))

def embedding_lookup(embeddings, index, sparse_input=True):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense()  for i in range(3)]
    else:
        return [embeddings[i][index[:, i], :]  for i in range(3)]

def read_data_sets(train_dir):
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    data = np.load(os.path.join(train_dir, TRAIN_FILE))
    train_data = DataSet(data['train_data'], data['nums_type'])
    labels = data['labels'] if 'labels' in data else None
    idx_label = data['idx_label'] if 'idx_label' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    del data
    data = np.load(os.path.join(train_dir, TEST_FILE))
    test_data = DataSet(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    embeddings = generate_embeddings(train_data.edge, train_data.nums_type) # 当前node type H与其他两个H 之间的邻接关系（HH_(other2)^T）
    # 再经过 列归一化x[:,i]/max(x[:,i] 得到
    return Datasets(train=train_data, test=test_data, embeddings=embeddings, node_cluster=node_cluster,
                labels=labels, idx_label=idx_label, label_name=label_set)

def generate_H(edge, nums_type):
    nums_examples = len(edge)
    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples)) for i in range(3)]
    return H

def dense_to_onehot(labels):
    return np.array(map(lambda x: [x*0.5+0.5, x*-0.5+0.5], list(labels)), dtype=float)

def generate_embeddings(edge, nums_type, H=None):
    if H is None:
        H = generate_H(edge, nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)] # dot 矩阵乘法
    # 得到当前node type H与其他两个H 之间的邻接关系
    ### 0-1 scaling ， 对每个值除于其对应列的最大值 x[:,i]/max(x[:,i])
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]
    return embeddings
