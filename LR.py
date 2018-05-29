#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    LR.py
  Author:       locke
  Date created: 2018/5/21 下午9:49
"""

import numpy as np
from sklearn import linear_model, metrics, preprocessing
from data_utils_cora import load_data, get_splits

np.random.seed(2018)

X, A, y = load_data(dataset='cora')
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits('cora', y)


METHOD = "line"
PARA = "_0.25_0.25" if METHOD == "node2vec" else ""
if METHOD == "line":
    PARA = "_2"

embeddings = np.genfromtxt("workspace/vec_2nd_wo_norm_100000.txt", skip_header=1, dtype=np.float32)[:, 1:]
embeddings = preprocessing.normalize(embeddings, axis=1)


# embeddings = np.load("workspace/"+METHOD+"_embedding_cora"+PARA+".npy")

# embeddings = np.load("workspace/line.tensorflow_16w.npy")

print(embeddings[0:2])

labels = np.where(y)[1]

model = linear_model.LogisticRegression(C=0.3, solver="liblinear", random_state=2018)
model.fit(embeddings[idx_train], labels[idx_train])
y_true = labels[idx_test]
y_pred = model.predict(embeddings[idx_test])
acc = metrics.accuracy_score(y_true, y_pred)
print("acc:", acc)
