#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    Node2Vec.py
  Author:       locke
  Date created: 2018/5/9 上午9:41
"""

import argparse
import time
import random
import numpy as np
from gensim.models import Word2Vec
from data_utils_cora import load_data

random.seed(2018)
np.random.seed(2018)


def alias_sampler(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def sampling(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def train(args):
    _, A, _ = load_data(path=args.path, dataset=args.dataset)
    row, col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build")
    t1 = time.time()
    G, node_samplers, edge_samplers = {}, {}, {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)
        node_samplers[node] = alias_sampler(probs=A[int(node), :].data / np.sum(A[int(node), :].data))

    for [i, j] in edges:
        edge_weights = []
        for j_nbr in G[j]:
            if j_nbr == i:
                edge_weights.append(A[int(j), int(j_nbr)] / args.p)
            elif A[int(j_nbr), int(i)] >= 1e-4:
                edge_weights.append(A[int(j), int(j_nbr)])
            else:
                edge_weights.append(A[int(j), int(j_nbr)] / args.q)
        edge_weights = np.asarray(edge_weights, dtype=np.float32)
        edge_samplers[i + "-" + j] = alias_sampler(probs=edge_weights / edge_weights.sum())

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = []
    for cnt in range(args.number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]
            while len(path) < args.walk_length:
                cur = path[-1]
                if len(G[cur]) > 0:
                    if len(path) == 1:
                        path.append(G[cur][sampling(node_samplers[cur][0], node_samplers[cur][1])])
                    else:
                        prev = path[-2]
                        path.append(
                            G[cur][sampling(edge_samplers[prev + "-" + cur][0], edge_samplers[prev + "-" + cur][1])])
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print("cost: {}s".format(t2 - t1))
    print("train...")
    model = Word2Vec(corpus, size=args.size, window=args.window, min_count=0, sg=1, workers=args.workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:
            output.append(model.wv[str(i)])
        else:
            output.append(np.zeros(args.size))
    np.save(args.output + "_" + str(args.p) + "_" + str(args.q) + ".npy", np.asarray(output, dtype=np.float32))
    print("saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/cora/', help='the input file of a network')
    parser.add_argument('--dataset', default='cora', help='the input file of a network')
    parser.add_argument('--output', default='workspace/node2vec_embedding_cora', help='the output file of the embedding')

    # parser.add_argument('--path', default='data/tencent/', help='the input file of a network')
    # parser.add_argument('--dataset', default='tencent', help='the input file of a network')
    # parser.add_argument('--output', default='workspace/node2vec_embedding_tencent', help='the output file of the embedding')

    parser.add_argument('--size', default=128, help='number of latent dimensions to learn for each node')
    parser.add_argument('--number_walks', default=10, help='number of random walks to start at each node')
    parser.add_argument('--walk_length', default=80, help='length of the random walk started at each node')
    parser.add_argument('--window', default=10, help='window size of skipgram model')
    parser.add_argument('--p', default=4, help='return hyperparameter, default is 1')
    parser.add_argument('--q', default=0.25, help='inout hyperparameter, default is 1')
    parser.add_argument('--workers', default=2, help='number of parallel processes')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
