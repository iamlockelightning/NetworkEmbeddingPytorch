#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    LRclassifier.py
  Author:       locke
  Date created: 2018/5/7 下午1:38
"""

import time
import numpy as np
from data_utils_cora import load_data, get_splits
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

np.random.seed(2018)
torch.manual_seed(2018)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        # out = F.relu(out)
        return F.log_softmax(out, dim=1)


X, A, y = load_data(dataset='cora')
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits('cora', y)


METHOD = "line"
PARA = "_4_0.25" if METHOD == "node2vec" else ""
if METHOD == "line":
    PARA = "_2"

# embeddings = np.genfromtxt("workspace/vec_2nd_wo_norm10000.txt", skip_header=1, dtype=np.float32)[:, 1:]
# from sklearn.preprocessing import normalize
# embeddings = normalize(embeddings, axis=1)


embeddings = np.load("workspace/"+METHOD+"_embedding_cora"+PARA+".npy")

# embeddings = np.load("workspace/line.tensorflow_7w.npy")
# print(embeddings[0:5])


embeddings = Variable(torch.FloatTensor(embeddings))
# print(embeddings)
labels = Variable(torch.LongTensor(np.where(y)[1]))
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

print("embeddings.shape: {}".format(embeddings.shape))
print("labels.shape: {}".format(labels.shape))
print("idx_train.shape: {}".format(idx_train.shape))
print("idx_val.shape: {}".format(idx_val.shape))
print("idx_test.shape: {}".format(idx_test.shape))

input_size = embeddings.shape[1]
num_classes = y_train.shape[1]
# learning_rate = 0.01
# weight_decay = 5e-4
# num_epochs = 300
# display_every = 30

learning_rate = 0.01
weight_decay = 5e-4
num_epochs = 1000
display_every = 100

model = LogisticRegression(input_size, num_classes)
print(model)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def accuracy(output, label):
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def train(epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()

    output = model(embeddings[idx_train])
    loss_train = F.nll_loss(output, labels[idx_train])
    acc_train = accuracy(output, labels[idx_train])

    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(embeddings[idx_val])
    loss_val = F.nll_loss(output, labels[idx_val])
    acc_val = accuracy(output, labels[idx_val])
    if epoch % display_every == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(embeddings[idx_test])
    loss_test = F.nll_loss(output, labels[idx_test])
    acc_test = accuracy(output, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
for epoch in range(num_epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
