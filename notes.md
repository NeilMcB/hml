# Hands-on Machine Learning

Notes from reading through [Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).

## Misc

* Obvious, but remember to consider how your algorithm scales with the number of features and the number of training instances!

## Chapter 2 - End-to-End Machine Learning Landscape

Scikit-Learn makes use of duck typing to define custom transformers/estimators/predictors, e.g.:

```python
class MyCustomTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, option):
        self.option = option
        
    def fit(self, X, y=None):
        # fit the transformer
        return self
    
    def transform(self, X):
        # transform the data
        return X_transformed
```
Adding `BaseEstimator` handles parameter access neatly and `TransformerMixin` gives us the
`fit_transform` method for free.


Using `sklearn.compose.ColumnTransformer` you can bring together lots of different transformations
applied to subsets of columns into one handy pipeline stage. This can be combined with e.g. models
in another pipeline wrapper:
```python
transformations = ColumnTransformer([
    ('numerical', numerical_transformation_pipeline, numerical_columns),
    ('categorical', categorical_transformation_pipeline, categorical_columns),
])

end_to_end_pipeline = Pipeline([
    ('transformations', transformations),
    ('model', MyFancyMLModel()),
])
```

This can be passed to e.g. `GridSearchCV` and the hyperparameters accessed as:
```python
{
    'model__n_estimators': [3, 10, 30],
    'transformations__numerical__my_numerical_parameter': [1, 2, 3],
}
```

## Chapter 3 - Classificaton

F_1 score is the harmonic mean of precision and recall. This heavily penalises low values, so will
only be high if both precision and recall are high.

Some Scikit-learn models support multliclass classification:
```python
model = KNeighborsClassifier()
model.fit(X_train, y_train)  # y.shappe == [N, K], k > 1
```

## Chapter 4 - Training Models

Scikit-learn includes a handy transformer for adding polynomial features:
```python
from sklearn.preprocessing import PolynomialFeatures
...
polynomial_features = PolynomialFeatures(degree=k, include_bias=True)
X_polynomial = polynomial_features.fit_transform(X)
```

Most regularised models required scaling - we want the coefficents for each feature (or combination
of features) to be penalised equally according to magnitude, otherwise large coefficients would be
penalised even if they were required due to the shape of the data.

### Types of regularisation:
* __Ridge__: L2 regularisation
  * Has a closed form solution
* __Least Absolute Shrinkage and Selection Operator (LASSO)__: L1 regularisation
  * Encourages sparse models - encourages complete elimination of unimportant features
  * May be unstable around the maximum - be careful with learning rate here
* __Elastic Net__: Combination of the two, with a mixing ratio defined as an extra hyperparameter

Early stopping is good.

### Softmax

We would describe softmax as "multiclass" - i.e. it can decide between mutually exclusive classes.
If we wanted to predict potentially multiple occurences within a single instance we would use what
we call a "multioutput" classifier.

We train softmax using cross-entropy - this is the multiclass generatlisation of log loss (i.e. we
sum over each p_k per training instance, instead of just p and p-1).

### Probability Distributions

Kullback-Leibler (KL) divergence measures the difference between two probability distributions.


## Chapter 5 - Support Vector Machines

For classification SVMs try to fit the widest possible "street" between the two classes of data,
whilst minimising the number of margin violations (few datasets are truly linearly separable). On
the other hand, for regression SVMs try to fit as much of the dataset onto the "street" itself.

SVM works best on small to medium sized complex datasets. It's sensitive to feature scales, so
use a `StandardScaler` first.

Generally we fit these using the _hinge_ loss function, more on this later.

There are various choices of kernel to consider:
* __Linear__: This should be the default to try; it's pretty quick to run. You can model non-linear boundaries using simple feature augmentation.
  * `c`: Controls for how many margin violations are permitted - i.e. how far into our clusters do we place the support vectors.
* __Polynomial__: This uses the kernel trick to fit non-linear boundaries automatically.
  * `c`: As above.
  * `degree`: The degree of polynomial features to fit.
  * `coef0`: How much to prefer/penalise high-degre features.
* __Gaussian Radial Basis Function (RBF)__: This places a spherical Gaussian kernel at each point in the dataset and fis a decision boundary based on these.
  * `c`: As above; the lower this is, the more "wiggly" the decision boundary will become.
  * `gamma`: This tells us how far the "influence" of each basis function travels.
* __and the rest...__: More basis functions exist, but are generally used for specialised purposes - e.g. string kernels.

Sklearn's SVC works particularly well when features are sparse.

We can use SVR (i.e. Support Vector _Regression_) models to identify outliers - these are the points that lie outside the margins of the SVM's "street".


## Chapter 6

A decision tree is _non-parametric_, this means:
* __non-parametric__: The model structure can freely adapt to the data.
* __parametric__: We have to choose ahead of time how many parameters are required to model the data, e.g. a linear model with a number of transformed features.

### Classification

Gini impurity measures the probability of each class $k$ being present in given node $i$:
$$G_i = 1 - \sum_{k=1}^Kp_{i,k}$$

"Pure" nodes have a Gini impurity of zero. Where $p_{i,k}$ is simply estimated by comparing the total number of training samples to the number of training samples whose features lead it to the node. Such probabilities can be used to estimate the confidince with which leaf nodes make predictions.

CART recursively tries to minimise the weighted (by number of samples passing to each) average Gini impurity of each pair of child nodes in a binary tree. It splits the training set in half by choosing the feature $k$ and threshold $t_k$ which minimises the quantity:
$$J(k,t_k) = \frac{m_l}{m}G_l + \frac{m_r}{m}G_r$$

Where $m$ is the total number of training samples, $m_l$ and $m_r$ are the number of instances passed to the left and right children respectively, and $G_l$ and $G_r$ are the corresponding Gini impurities.

This is repeated recursively until any one of a number of possible conditions are met:
* The leaf node is pure.
* `max_depth`: This path of the tree has reached a specified maximum depth.
* `min_samples_split`: When the number of samples passed to a node falls below this number, the node will be declared a leaf node and thus no further splits will occur. 
* `min_samples_leaf`: When a split is made on a parent node, if the number of samples that pass through to its child is below this number then the split will not take place.
* ... and more

### Regression

CART can also be used to produce regression trees. The prediction made is the average label value of all the training samples that reach a given node in the tree. The adapted card algorithm recursively tries to minimise:
$$J(k,t_k) = \frac{m_l}{m}\mathrm{MSE}_l + \frac{m_r}{m}\mathrm{MSE}_r$$
Where:
$$\mathrm{MSE}_j = \sum_{i\in j}(\hat{y}_j - y^{(i)})$$
and
$$\hat{y}_j = \frac{1}{m_j}\sum_{i\in j}y^{(i)}$$

### General

Decision trees give orthogonal decision boundaries so can be sensitive to dataset rotation.