#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 01:46:17 2018

@author: stremler

source: sklearn/preprocessing/data.py
"""

from itertools import combinations_with_replacement as combinations_w_r
from itertools import chain
import numpy as np

class PolynomialFeatures:
    """PolynomialFeatures"""
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        
    def fit(self, X):
        n_samples, n_features = X.shape
        combinations = self._combinations(n_features, self.degree,
                                          self.include_bias)
        self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self
        
    def transform(self, X):
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)

        combinations = self._combinations(n_features, self.degree,
                                          self.include_bias)
        for i, c in enumerate(combinations):
            XP[:, i] = X[:, c].prod(1)

        return XP
        
    def fit_transform(self, X):
        return self, self.fit(X).transform(X)
        
    def _combinations(self, n_features, degree, include_bias):
        start = int(not include_bias)
        return chain.from_iterable(combinations_w_r(range(n_features), i)
                                   for i in range(start, degree + 1))