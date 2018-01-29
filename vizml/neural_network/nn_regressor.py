#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:57:15 2018

@author: stremler
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class NNRegressor:
    """NNRegressor"""
    def __init__(self, eta=0.001, layers=2, epoch=100):
        self.eta = eta
        self.layers = np.atleast_1d(layers)
        self.epoch = epoch
    
    def fit(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.atleast_2d(y).T
        self.layers = np.concatenate(([self.X.shape[1]], self.layers, [self.y.shape[1]]))
        self.w = self._initialize_weights(self.layers)
        self.weight_steps = [self.w]
        
    def _initialize_weights(self, layers):
        w = []
        
        for i in range(len(layers) - 1):
            # number of neurons in the i-th and i+1-th layer
            n_no = (layers[i] + 1) + layers[i + 1]
            
            w.append(np.random.uniform(low=-np.sqrt(6/n_no), high=np.sqrt(6/n_no), 
                              size=(layers[i] + 1, layers[i + 1])))
            
        return w