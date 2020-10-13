# XGBoost

Original paper: https://arxiv.org/pdf/1603.02754.pdf

XGBoost stands for eXtreme Gradient Boosting

## Weighted Quantile Sketch

One important step in the approximate algorithm is to
propose candidate split points.
We introduced a novel distributed
weighted quantile sketch algorithm that can handle weighted
data with a provable theoretical guarantee.

### Model Features

Three main forms of gradient boosting are supported:

- **Gradient Boosting** algorithm also called gradient boosting machine including the learning rate.
- **Stochastic Gradient Boosting** with sub-sampling at the row, column and column per split levels.
- **Regularized Gradient Boosting** with both L1 and L2 regularization.

### System Features

The library provides a system for use in a range of computing environments, not least:

- **Parallelization** of tree construction using all of your CPU cores during training.
- **Distributed Computing** for training very large models using a cluster of machines.
- **Out-of-Core Computing** for very large datasets that don’t fit into memory.
- **Cache Optimization** of data structures and algorithm to make best use of hardware
