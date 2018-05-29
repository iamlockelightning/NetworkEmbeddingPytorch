#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    GraRep.py
  Author:       locke
  Date created: 2018/5/9 上午9:57
"""

import argparse
import time
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
from data_utils_cora import load_data

import torch
import torch.nn.functional as F

np.random.seed(2018)
torch.manual_seed(2018)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2018)
else:
    print("Device no gpu...")


def train_py(args):
    _, A, _ = load_data(path=args.path, dataset=args.dataset)

    scaled_A = A / A.sum(axis=1)
    size = args.size
    K = args.Kstep
    assert size % K == 0
    dim = int(size / K)
    t1 = time.time()
    A_k = np.identity(scaled_A.shape[0])
    Rep = np.zeros((scaled_A.shape[0], size))

    scaled_A = torch.FloatTensor(scaled_A).cuda()
    A_k = torch.FloatTensor(A_k).cuda()
    Rep = torch.FloatTensor(Rep).cuda()
    for i in range(K):
        print("K:", i)
        A_k = torch.dot(A_k, scaled_A)
        prob_trans = torch.log(A_k / torch.sum(A_k, dim=0).repeat(scaled_A.shape[0], 1)) - torch.log(
            1.0 / scaled_A.shape[0])
        prob_trans[prob_trans < 0] = 0
        prob_trans[prob_trans == np.nan] = 0
        U, S, VT = torch.svd(prob_trans)
        Ud = U[:, 0:dim]
        Sd = S[0:dim]
        R_k = Ud * torch.pow(Sd, 0.5).view(dim)
        R_k = F.normalize(R_k, p=2, dim=1)
        Rep[:, dim * i: dim * (i + 1)] = R_k[:, :]
    print("done.., cost: {}s".format(time.time() - t1))
    np.save(args.output + ".npy", Rep.cpu().numpy())
    print("saved.")


def train(args):
    _, A, _ = load_data(path=args.path, dataset=args.dataset)
    scaled_A = A / A.sum(axis=1)
    size = args.size
    K = args.Kstep
    assert size % K == 0
    dim = int(size / K)
    t1 = time.time()
    A_k = np.identity(scaled_A.shape[0])
    Rep = np.zeros((scaled_A.shape[0], size))
    for i in range(K):
        print("K:", i)
        A_k = np.dot(A_k, scaled_A)
        prob_trans = np.log(A_k / np.tile(np.sum(A_k, axis=0), (scaled_A.shape[0], 1))) - np.log(
            1.0 / scaled_A.shape[0])
        prob_trans[prob_trans < 0] = 0
        prob_trans[prob_trans == np.nan] = 0
        U, S, VT = la.svd(prob_trans)
        Ud = U[:, 0:dim]
        Sd = S[0:dim]
        R_k = np.array(Ud) * np.power(Sd, 0.5).reshape(dim)
        R_k = normalize(R_k, axis=1, norm='l2')
        Rep[:, dim * i: dim * (i + 1)] = R_k[:, :]
    print("done.., cost: {}s".format(time.time() - t1))
    np.save(args.output + ".npy", np.asarray(Rep, dtype=np.float32))
    print("saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/cora/', help='the input file of a network')
    parser.add_argument('--dataset', default='cora', help='the input file of a network')
    parser.add_argument('--output', default='workspace/grarep_embedding_cora', help='the output file of the embedding')

    # parser.add_argument('--path', default='data/tencent/', help='the input file of a network')
    # parser.add_argument('--dataset', default='tencent', help='the input file of a network')
    # parser.add_argument('--output', default='workspace/grarep_embedding_tencent', help='the output file of the embedding')

    parser.add_argument('--size', default=128, help='number of latent dimensions to learn for each node')
    parser.add_argument('--Kstep', default=4, help='use k-step transition probability matrix')
    args = parser.parse_args()
    print(args)
    # train(args)
    train_py(args)

if __name__ == '__main__':
    main()
