# Hands-on Machine Learning

Notes from reading through [Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).

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