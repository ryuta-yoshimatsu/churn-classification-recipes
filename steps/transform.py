"""
This module defines the following routines used by the 'transform' step:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from pandas import DataFrame
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


import logging
_logger = logging.getLogger(__name__)


def calculate_features(df: DataFrame):
    """
    Extend the input dataframe
    """
    return df


def transformer_fn(ohe: bool = False):
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.
    # Convert label to int and rename column
    _logger.info(f'Transforming data')

    if not ohe:
        _logger.info("ohe set to false")

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer',
             SimpleImputer(strategy='median'),
             make_column_selector(dtype_exclude='object')
             ),
            ('categorical_transformer',
             OneHotEncoder(handle_unknown='ignore'),
             make_column_selector(dtype_include='object')
             ),
        ],
        remainder='passthrough',
        sparse_threshold=0
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return pipeline

