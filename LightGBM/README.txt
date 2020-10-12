Source: http://www.audentia-gestion.fr/MICROSOFT/lightgbm.pdf

Gradient Boosting Decision Tree (GBDT)
Problem: The efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large.
The most time-consuming part in learning a decision tree is to find the best split points.
Solution: -> LightGBM

Using two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).
With histogram-based algorithm, to regroup continuous feature values into discrete bins and uses these
bins to construct feature histograms during training.

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
