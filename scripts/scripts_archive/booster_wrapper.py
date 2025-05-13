# scripts/booster_wrapper.py
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class BoosterWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster, feature_names):
        """
        booster:       an xgb.Booster (from xgb.train)
        feature_names: list of column names in the exact order you trained on
        """
        self.booster = booster
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # just record the classes, nothing else
        # X can be a DataFrame or array with exactly those feature_names
        self.classes_ = np.unique(y)
        # optional but sometimes useful
        self.n_features_in_ = len(self.feature_names)
        return self

    def predict_proba(self, X):
        # build a DMatrix from the exact feature columns
        # if X is a DataFrame:
        df = X if hasattr(X, "loc") else pd.DataFrame(X, columns=self.feature_names)
        dmat = xgb.DMatrix(df[self.feature_names], feature_names=self.feature_names)
        p1 = self.booster.predict(dmat)               # returns 1-d array
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
