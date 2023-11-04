# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月30日
"""

import scipy
import numpy as np

def compute_nodes_distance(X_view):
    """
    计算node之间的距离
    :param X_view: [n_nodes,features]
    :return:
        dis: [n_nodes,n_nodes]
    """
    dis = np.power(X_view[:, np.newaxis, :] - X_view[np.newaxis, :, :],2) #[n,n,f]
    dis = np.sum(dis,axis=-1) #[n,n]
    dis = np.sqrt(dis) #[n,n]
    return dis

def get_adj_matrix(X_view, topk):
    """
    get adjacent matrix from x_veiw with knn method.
    :param X_view: [n_nodes, n_features]
    :param topk: int
    :return:
        A_: #[n_nodes,n_nodes] 规范化的邻接矩阵
    """
    #计算节点之间的距离
    dis = compute_nodes_distance(X_view) #[n,n]

    #从小到大排序，获得下标
    argsorted = np.argsort(dis,axis=-1)

    #选择前topk个节点
    j_args = argsorted[:,:topk] #[n,topk]
    i_args = np.repeat(np.arange(j_args.shape[0])[:,np.newaxis],topk,axis=1) #[n,topk]

    #A + I
    A_I = np.zeros_like(dis) #[n,n]
    A_I[i_args,j_args] = 1 #[n,n]

    #D^(-1/2)
    D_1_2 = np.diag(A_I.sum(1)**(-0.5)) #[n,n]
    A_ = D_1_2 @ A_I @ D_1_2 #[n,n]

    return A_

def inspect_multiview_features_shape(features_list):
    """
    print multiview inputs data shape
    :param features_list: [n_view, n_node, n_features]
    :return:
        None
    """
    for i in range(len(features_list)):
        print(features_list[i].shape)

def inspect_multiview_labels(labels):
    print(labels.shape)
    print(labels.max())
    print(labels.min())

a = scipy.io.loadmat('../data/animals.mat')
X = a['X'][0]
Y = a['Y']

inspect_multiview_features_shape(X)
inspect_multiview_labels(Y)
