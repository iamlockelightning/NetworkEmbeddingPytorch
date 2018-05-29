#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    DeepWalk.py
  Author:       locke
  Date created: 2018/5/8 上午10:03
"""

import argparse
import time
import random
import numpy as np
from gensim.models import Word2Vec
from data_utils_cora import load_data

random.seed(2018)
np.random.seed(2018)


def train(args):
    _, A, _ = load_data(path=args.path, dataset=args.dataset)
    row, col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build")
    t1 = time.time()
    G = {}
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
                    if random.random() >= args.alpha:
                        path.append(random.choice(G[cur]))
                    else:
                        path.append(path[0])
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print("cost: {}s".format(t2 - t1))
    print("train...")
    model = Word2Vec(corpus, size=args.size, window=args.window, min_count=0, sg=1, hs=1, workers=args.workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:
            output.append(model.wv[str(i)])
        else:
            output.append(np.zeros(args.size))
    np.save(args.output + ".npy", np.asarray(output, dtype=np.float32))
    print("saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/cora/', help='the input file of a network')
    parser.add_argument('--dataset', default='cora', help='the input file of a network')
    parser.add_argument('--output', default='workspace/deepwalk_embedding_cora',
                        help='the output file of the embedding')

    # parser.add_argument('--path', default='data/tencent/', help='the input file of a network')
    # parser.add_argument('--dataset', default='tencent', help='the input file of a network')
    # parser.add_argument('--output', default='workspace/deepwalk_embedding_tencent', help='the output file of the embedding')

    parser.add_argument('--size', default=128, help='number of latent dimensions to learn for each node')
    parser.add_argument('--number_walks', default=10, help='number of random walks to start at each node')
    parser.add_argument('--walk_length', default=80, help='length of the random walk started at each node')
    parser.add_argument('--window', default=10, help='window size of skipgram model')
    parser.add_argument('--alpha', default=0, help='jump probability')
    parser.add_argument('--workers', default=2, help='number of parallel processes')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
