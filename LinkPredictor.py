#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    LinkPredictor.py
  Author:       locke
  Date created: 2018/5/7 下午8:03
"""

import numpy as np
from data_utils_cora import get_splits
from sklearn import metrics

np.random.seed(2018)

METHOD = "line"
PARA = "_4_0.25" if METHOD == "node2vec" else  ""

# embeddings = np.load("workspace/"+METHOD+"_embedding_tencent"+PARA+".npy")

embeddings = np.load("workspace/line.tensorflow_16w.npy") # 0.4796124739071113
print(embeddings[0:2])

print("embeddings.shape: {}".format(embeddings.shape))
test_edges_true, test_edges_false = get_splits(typ="tencent", y="data/tencent/")
print("test_edges_true.shape: {}".format(test_edges_true.shape))
print("test_edges_false.shape: {}".format(test_edges_false.shape))

y, pred = [], []
for i, typ_edges in enumerate([test_edges_true, test_edges_false]):
    for edge in typ_edges:
        pred.append((metrics.pairwise.cosine_similarity([embeddings[edge[0], :]], [embeddings[edge[1], :]])[0][0] + 1.0) / 2.0)
        y.append(1-i)

y = np.array(y)
pred = np.array(pred)
print(y)
print(pred)

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

print("thres: {}, auc: {}".format(thresholds, metrics.auc(fpr, tpr)))


test_auc = metrics.roc_auc_score(y, pred)
print("auc: {}".format(test_auc))
