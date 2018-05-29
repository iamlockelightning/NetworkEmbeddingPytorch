#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    LINE.py
  Author:       locke
  Date created: 2018/5/6 下午4:58
"""

import argparse
import numpy as np
from data_utils_cora import load_data
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(2018)
torch.manual_seed(2018)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2018)
else:
    print("Device no gpu...")


# CUDA_VISIBLE_DEVICES=3 python3 run.py

class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, probs):
        self.n = len(probs)
        self.U = np.array(probs) * self.n
        self.K = [i for i in range(len(probs))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


def get_batch(A, edges, edge_sampler, node_sampler, batch_size, negative):
    edge_batch_index = edge_sampler.sampling(batch_size)
    u_i, u_j, label = [], [], []
    for edge_index in edge_batch_index:
        edge = edges[edge_index]
        if np.random.rand() > 0.5:  # ?? important: second-order proximity is for directed edge
            edge = (edge[1], edge[0])
        u_i.append(edge[0])
        u_j.append(edge[1])
        label.append(1)
        for i in range(negative):
            while True:
                negative_node = node_sampler.sampling()
                if A[negative_node, edge[1]].data[0] <= 1e-4:
                    break
            u_i.append(edge[0])
            u_j.append(negative_node)
            label.append(-1)
    u_i = np.array(u_i, dtype=np.int32)
    u_j = np.array(u_j, dtype=np.int32)
    label = np.array(label, dtype=np.int32)
    u_i = torch.LongTensor(u_i)
    u_j = torch.LongTensor(u_j)
    label = torch.FloatTensor(label)
    return u_i, u_j, label


class Line(nn.Module):
    def __init__(self, node_size, emb_size, order=1):
        super(Line, self).__init__()
        self.order = order
        self.embeddings = nn.Embedding(node_size, emb_size)
        if self.order == 2:
            self.context_embedding = nn.Embedding(node_size, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal(self.embeddings.weight.data, mean=0, std=0.01)
        if self.order == 2:
            nn.init.normal(self.context_embedding.weight.data, mean=0, std=0.01)

    def forward(self, u_i, u_j, label):
        emb_u_i = self.embeddings(u_i)
        if self.order == 1:
            emb_u_j = self.embeddings(u_j)
        else:  # self.order == 2:
            emb_u_j = self.context_embedding(u_j)
        inner_product = torch.sum(emb_u_i * emb_u_j, dim=1)
        loss = - torch.mean(F.logsigmoid(label * inner_product))
        return loss


def train(args):
    _, A, _ = load_data(path=args.path, dataset=args.dataset)
    row, col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1)
    edge_sampler = AliasSampling(probs=A.data / np.sum(A.data))
    node_weights = np.power(np.asarray(A.sum(axis=0)).flatten(), 0.75)
    node_sampler = AliasSampling(probs=node_weights / np.sum(node_weights))

    learning_rate = args.rho
    line = Line(A.shape[0], args.size)
    optimizer = optim.Adadelta(line.parameters(), lr=learning_rate)
    if args.gpu and torch.cuda.is_available():
        line.cuda()

    sampling_time, training_time = 0, 0
    line.train()
    for i in range(args.batch_num):
        t1 = time.time()
        u_i, u_j, label = get_batch(A, edges=edges, edge_sampler=edge_sampler, node_sampler=node_sampler,
                                    batch_size=args.batch_size, negative=args.negative)
        t2 = time.time()
        sampling_time += t2 - t1

        if args.gpu and torch.cuda.is_available():
            u_i, u_j, label = Variable(u_i.cuda()), Variable(u_j.cuda()), Variable(label.cuda())
        else:
            u_i, u_j, label = Variable(u_i), Variable(u_j), Variable(label)
        if i % 100 == 0 and i != 0:
            print('Batch_no: {:06d}'.format(i),
                  'loss: {:.4f}'.format(loss.data[0]),
                  'rho: {:.4f}'.format(learning_rate),
                  'sampling_time: {:.4f}'.format(sampling_time),
                  'training_time: {:.4f}'.format(training_time))
            sampling_time, training_time = 0, 0
        else:
            optimizer.zero_grad()

            loss = line(u_i, u_j, label)
            # loss = F.kl_div(output, label)
            # print("__loss: {:.4f}".format(loss.data[0]))

            loss.backward()

            # print("line.embeddings.weight.grad:", np.max(np.array(line.embeddings.weight.grad.data)))
            # if line.order == 2:
            #     print("line.context_embedding.weight.grad:", np.max(np.array(line.context_embedding.weight.grad.data)))

            optimizer.step()

            training_time += time.time() - t2

            if learning_rate > args.rho * 1e-4:
                learning_rate = args.rho * (1 - i / args.batch_num)
            else:
                learning_rate = args.rho * 1e-4
            optimizer = optim.Adadelta(line.parameters(), lr=learning_rate)

    print("done..")
    if args.gpu and torch.cuda.is_available():
        np.save(args.output + "_" + str(args.order) + ".npy", F.normalize(line.embeddings.cpu().weight).data.numpy())
    else:
        np.save(args.output + "_" + str(args.order) + ".npy", F.normalize(line.embeddings.weight).data.numpy())
    print("saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/cora/', help='the input file of a network')
    parser.add_argument('--dataset', default='cora', help='the input file of a network')
    parser.add_argument('--output', default='workspace/line_embedding_cora', help='the output file of the embedding')

    # parser.add_argument('--path', default='data/tencent/', help='the input file of a network')
    # parser.add_argument('--dataset', default='tencent', help='the input file of a network')
    # parser.add_argument('--output', default='workspace/line_embedding_tencent', help='the output file of the embedding')

    parser.add_argument('--size', default=128, help='the dimension of the embedding')
    parser.add_argument('--order', default=2, help='the order of the proximity, 1 for first order, 2 for second order')
    parser.add_argument('--negative', default=5, help='the number of negative samples used in negative sampling')
    parser.add_argument('--batch_num', default=50000, help='the total number of training batch num')
    parser.add_argument('--batch_size', default=100, help='the total number of training batch size')
    parser.add_argument('--rho', default=0.025, help='the starting value of the learning rate')
    parser.add_argument('--gpu', default=True, help='whether to use GPU')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
