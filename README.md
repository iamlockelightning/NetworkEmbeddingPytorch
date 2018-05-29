# NetworkEmbeddingPytorch
Pytorch implementation of some Network Embedding methods.


## For Cora and Tencent dataset

### Methods
- DeepWalk
- LINE
- Node2Vec
- GraRep

### Train
Change the arguments in the scripts before runing the code.

- DeepWalk
```Python
python3 DeepWalk.py
```

- LINE
```Python
python3 LINE.py
```

- Node2Vec
```Python
python3 Node2Vec.py
```

- GraRep
```Python
python3 GraRep.py
```

### Test
- Logistic Regression Classification
```Python
python3 LRclassifier.py
(python3 LR.py)
```

- Link Prediction
```Python
python3 LinkPredictor.py
```

## Requirements
- Python (3.5.2)
- PyTorch (0.3.0)

## Reference and Citing
Implementation refers to some public implementations in other languages or other frameworks:
- DeepWalk: Python[https://github.com/phanein/deepwalk], 
- LINE: C/C++[https://github.com/tangjianpku/LINE], Python/TensorFlow[https://github.com/snowkylin/line]
- node2vec: Python[https://github.com/aditya-grover/node2vec], Python[https://github.com/thunlp/OpenNE]
- GraRep: Matlab[https://github.com/ShelsonCao/GraRep], Python[https://github.com/thunlp/OpenNE]

If you find the implementation useful in your research, please cite the following papers:
- DeepWalk: Online Learning of Social Representations. Bryan Perozzi, Rami Al-Rfou, Steven Skiena. KDD 2014. 
- LINE: Large-scale Information Network Embedding. Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Me. WWW 2015.
- node2vec: Scalable Feature Learning for Networks. Aditya Grover, Jure Leskovec. KDD 2016.
- GraRep: Learning Graph Representations with Global Structural Information. Shaosheng Cao, Wei Lu, Qiongkai Xu. CIKM 2015. 

## Author
You're free to use the implementation. But if you find any bug, please kindly let me know in Issues. Thanks!
