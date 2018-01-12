#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 03:09:46 2018

@author: stremler
"""

import numpy as np

class StandardScaler:
    """StandardScaler"""
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self, self.fit(X).transform(X)