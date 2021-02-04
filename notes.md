# Hands-on Machine Learning

Notes from reading through [Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).

## Misc

* Obvious, but remember to consider how your algorithm scales with the number of features and the number of training instances!

## Chapter 2

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

## Chapter 3

F_1 score is the harmonic mean of precision and recall. This heavily penalises low values, so will
only be high if both precision and recall are high.

Some Scikit-learn models support multliclass classification:
```python
model = KNeighborsClassifier()
model.fit(X_train, y_train)  # y.shappe == [N, K], k > 1
```

## Chapter 4

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

## Chapter 5

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