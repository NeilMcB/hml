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
Adding `BaseEstimator` handles parameter access neatly and `TransformerMixin` gives us the `fit_transform` method for free.


Using `sklearn.compose.ColumnTransformer` you can bring together lots of different transformations applied to subsets of columns into one
handy pipeline stage. This can be combined with e.g. models in another pipeline wrapper:
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
{'model__n_estimators': [3, 10, 30], 'transformations__numerical__my_numerical_parameter': [1, 2, 3]}
```

