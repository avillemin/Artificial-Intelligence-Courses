# LightGBM

Source: http://www.audentia-gestion.fr/MICROSOFT/lightgbm.pdf

**Gradient Boosting Decision Tree** (GBDT)   
Problem: The efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large.
The most time-consuming part in learning a decision tree is to find the best split points.
Solution: -> LightGBM

Using two novel techniques: **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)**.
With **histogram-based algorithm**, to **regroup continuous feature values into discrete bins** and uses these
bins to construct feature histograms during training.

![alt text](https://slideplayer.com/slide/17648091/105/images/53/LGBM+Stands+for+Light+Gradient+Boosted+Machines.+It+is+a+library+for+training+GBMs+developed+by+Microsoft%2C+and+it+competes+with+XGBoost..jpg)

Gradient-based One-Side Sampling (GOSS). While there is no native weight for data instance in
GBDT, we notice that data instances with different gradients play different roles in the computation
of information gain. In particular, according to the definition of information gain, those instances
with larger gradients1
(i.e., under-trained instances) will contribute more to the information gain.
Therefore, when down sampling the data instances, in order to retain the accuracy of information gain
estimation, we should better keep those instances with large gradients (e.g., larger than a pre-defined
threshold, or among the top percentiles), and only randomly drop those instances with small gradients.
We prove that such a treatment can lead to a more accurate gain estimation than uniformly random
sampling, with the same target sampling rate, especially when the value of information gain has a
large range.

Exclusive Feature Bundling (EFB). Usually in real applications, although there are a large number
of features, the feature space is quite sparse, which provides us a possibility of designing a nearly
lossless approach to reduce the number of effective features. Specifically, in a sparse feature space,
many features are (almost) exclusive, i.e., they rarely take nonzero values simultaneously. Examples
include the one-hot features (e.g., one-hot word representation in text mining). We can safely bundle
such exclusive features. To this end, we design an efficient algorithm by reducing the optimal
bundling problem to a graph coloring problem (by taking features as vertices and adding edges for
every two features if they are not mutually exclusive), and solving it by a greedy algorithm with a
constant approximation ratio.

We call the new GBDT algorithm with GOSS and EFB LightGBM2.

The main cost in GBDT lies in learning the decision trees, and the most time-consuming part in
learning a decision tree is to find the best split points. One of the most popular algorithms to find split
points is the pre-sorted algorithm, which enumerates all possible split points on the pre-sorted
feature values. This algorithm is simple and can find the optimal split points, however, it is inefficient
in both training speed and memory consumption. Another popular algorithm is the histogram-based
algorithm. Instead of finding the split points on the sorted feature
values, histogram-based algorithm buckets continuous feature values into discrete bins and uses these
bins to construct feature histograms during training. Since the histogram-based algorithm is more
efficient in both memory consumption and training speed, we will develop our work on its basis.

The histogram-based algorithm finds the best split points based on the feature
histograms. It costs O(#data × #feature) for histogram building and O(#bin × #feature) for
split point finding. Since #bin is usually much smaller than #data, histogram building will dominate
the computational complexity. If we can reduce #data or #feature, we will be able to substantially
speed up the training of GBDT.

XGBoost supports both the pre-sorted algorithm and histogram-based algorithm.

To reduce the size of the training data, a common approach is to down sample the data instances. Similarly, to reduce the number of features, it is natural to filter weak features. However, these approaches
highly rely on the assumption that features contain significant redundancy, which might not always be true in practice.


##  Gradient-based One-Side Sampling

We propose a novel sampling method for GBDT that can achieve a good balance
between reducing the number of data instances and keeping the accuracy for learned decision trees.

If an instance is associated
with a small gradient, the training error for this instance is small and it is already well-trained.
A straightforward idea is to discard those data instances with small gradients. However, the data
distribution will be changed by doing so, which will hurt the accuracy of the learned model. To avoid
this problem, we propose a new method called Gradient-based One-Side Sampling (GOSS).

GOSS keeps all the instances with large gradients and performs random sampling on the instances with small gradients. In order to compensate the influence to the data distribution, when computing the
information gain, GOSS introduces a constant multiplier for the data instances with small gradients. Specifically, GOSS firstly sorts the data instances according to the absolute value of their
gradients and selects the top a×100% instances. Then it randomly samples b×100% instances from
the rest of the data. After that, GOSS amplifies the sampled data with small gradients by a constant
(1−a)/b when calculating the information gain. By doing so, we put more focus on the under-trained
instances without changing the original data distribution by much.

![alt text](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile10.uf.tistory.com%2Fimage%2F99BA2D3A5B54B00708F292)

## Exclusive Feature Bundling

-> novel method to effectively reduce the number of features

High-dimensional data are usually very sparse. The sparsity of the feature space provides us a
possibility of designing a nearly lossless approach to reduce the number of features. Specifically, in
a sparse feature space, many features are mutually exclusive, i.e., they never take nonzero values
simultaneously. We can safely bundle exclusive features into a single feature (which we call an
exclusive feature bundle). By a carefully designed feature scanning algorithm, we can build the
same feature histograms from the feature bundles as those from individual features. In this way, the
complexity of histogram building changes from O(#data × #feature) to O(#data × #bundle),
while #bundle << #feature. Then we can significantly speed up the training of GBDT without
hurting the accuracy.

There are two issues to be addressed. The first one is to determine which features should be bundled
together. The second is how to construct the bundle.

The solution is going to use graph coloring to partition features into a smallest number of exclusive bundles.

https://en.wikipedia.org/wiki/Graph_coloring
In graph theory, graph coloring is a special case of graph labeling; it is an assignment of labels traditionally called "colors" to elements of a graph subject to certain constraints. In its simplest form, it is a way of coloring the vertices of a graph such that no two adjacent vertices are of the same color; this is called a vertex coloring.
![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Petersen_graph_3-coloring.svg/220px-Petersen_graph_3-coloring.svg.png)


In order to
find a good approximation algorithm, we first reduce the optimal bundling problem to the graph
coloring problem by taking features as vertices and adding edges for every two features if they are
not mutually exclusive, then we use a greedy algorithm which can produce reasonably good results (with a constant approximation ratio) for graph coloring to produce the bundles. Furthermore, we
notice that there are usually quite a few features, although not 100% mutually exclusive, also rarely
take nonzero values simultaneously. If our algorithm can allow a small fraction of conflicts, we can
get an even smaller number of feature bundles and further improve the computational efficiency.

The time complexity is O(#feature²) and it is processed only once before training. This complexity is acceptable when the
number of features is not very large, but may still suffer if there are millions of features. To further
improve the efficiency, we propose a more efficient ordering strategy without building the graph:
ordering by the count of nonzero values, which is similar to ordering by degrees since more nonzero
values usually leads to higher probability of conflicts.

We need a good way of merging the features in the same bundle in order to
reduce the corresponding training complexity. The key is to ensure that the values of the original
features can be identified from the feature bundles. Since the histogram-based algorithm stores
discrete bins instead of continuous values of the features, we can construct a feature bundle by letting
exclusive features reside in different bins. This can be done by adding offsets to the original values of
the features.

## Gradient Boosting methods

https://towardsdatascience.com/understanding-lightgbm-parameters-and-how-to-tune-them-6764e20c6e5b
With LightGBM you can run different types of Gradient Boosting methods. You have: GBDT, DART, and GOSS which can be specified with the “boosting“ parameter.

### GBDT

This method is the **traditional Gradient Boosting Decision Tree** that was first suggested in this article and is the algorithm behind some great libraries like XGBoost and pGBRT.
It is based on three important principles:
- Weak learners (decision trees)
- Gradient Optimization
- Boosting Technique

So in the gbdt method we have a lot of decision trees(weak learners). Those trees are built sequentially:

- first tree learns how to fit to the target variable
- second tree learns how to fit to the residual (difference) between the predictions of the first tree and the ground truth
- The third tree learns how to fit the residuals of the second tree and so on.

**All those trees are trained by propagating the gradients of errors throughout the system**.
The main drawback of gbdt is that finding the best split points in each tree node is **time-consuming** and **memory-consuming** operation other boosting methods try to tackle that problem.

### DART

https://arxiv.org/pdf/1505.01866.pdf
-> Method that uses **dropout**, standard in Neural Networks, to improve model regularization and deal with some other less-obvious problems.
Gbdt suffers from over-specialization, which means trees added at later iterations tend to impact the prediction of only a few instances and make a negligible contribution towards the remaining instances. **Adding dropout makes it more difficult for the trees at later iterations to specialize on those few samples and hence improves the performance**.

![alt text](https://www.researchgate.net/profile/Ran_Gilad-Bachrach/publication/276149305/figure/fig2/AS:669315888070698@1536588751951/The-average-contribution-of-the-trees-in-the-ensemble-for-different-learning-algorithms_Q320.jpg)

DART diverges from MART at two places. First, when computing the
gradient that the next tree will fit, only a random subset of the existing ensemble is considered. The second place is when adding the new tree to the ensemble where
DART performs a normalization step. DART scales the new tree T by a factor of
1/k such that it will have the same order of magnitude
as the dropped trees, with k the number of trees dropped.   
On
one extreme, if no tree is dropped, DART is no different than MART. On the other extreme, if all the trees
are dropped, the DART is no different than random
forest.

### GOSS

Described above.   
GOSS suggests a **sampling method based on the gradient** to avoid searching for the whole search space

### Which one is the best ?

![alt text](https://miro.medium.com/max/587/0*TyyFwfF5AM5M_x2T.png)

## Histogram based decision tree

Source: https://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/   

The amount of time it takes to build a tree is proportional to the number of splits that have to be evaluated. Often, small changes in the split don't make much of a difference in the performance of the tree. Histogram-based methods take advantage of this fact by grouping features into a set of bins and perform splitting on the bins instead of the features. This is equivalent to subsampling the number of splits that the model evaluates. Since the features can be binned before building each tree, this method can greatly speed up training, reducing the computational complexity to O(n_{data} n_{bins}).   
Though conceptually simple, histogram-based methods present several choices that the user has to make. Firstly the number of bins creates a trade-off between speed and accuracy: the more bins there are, the more accurate the algorithm is, but the slower it is as well. Secondly, how to divide the features into discrete bins is a non-trivial problem: dividing the bins into equal intervals (the most simple method) can often result in an unbalanced allocation of data.   

## Ignoring sparse inputs (xgboost and lightGBM)

Xgboost and lightGBM tend to be used on tabular data or text data that has been vectorized. Therefore, the inputs to xgboost and lightGBM tend to be sparse. Since the vast majority of the values will be 0, **having to look through all the values of a sparse feature is wasteful**. Xgboost proposes to ignore the 0 features when computing the split, then allocating all the data with missing values to whichever side of the split reduces the loss more. This reduces the number of samples that have to be used when evaluating each split, speeding up the training process.   
Incidentally, xgboost and lightGBM both treat missing values in the same way as xgboost treats the zero values in sparse matrices; **it ignores them during split finding, then allocates them to whichever side reduces the loss the most**. 
Though lightGBM does not enable ignoring zero values by default, it has an option called zero_as_missing which, if set to True, will regard all zero values as missing. According to this thread on GitHub, lightGBM will treat missing values in the same way as xgboost as long as the parameter use_missing is set to True (which is the default behavior).

