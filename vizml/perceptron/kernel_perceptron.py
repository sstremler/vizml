#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 00:55:52 2018

@author: stremler
"""

import numpy as np
import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib import pyplot as plt
from matplotlib import animation

class KernelPerceptron:
    """KernelPerceptron"""
    def __init__(self, kernel="gaussian", kernel_param=1):
        if kernel == "gaussian":
            self.kernel = self._gaussian_kernel
        else: 
            self.kernel = self._polynomial_kernel
            
        self.kernel_param = kernel_param
        
    def fit(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = y
        self.alpha = np.zeros(self.X.shape[0])
        self.alpha_steps = self.alpha
        kernel_m = self.kernel(self.X, self.kernel_param)
        
        while True:
            error = False
            
            for i in range(X.shape[0]):
                if y[i] * np.dot(self.alpha * y, kernel_m[i,]) <= 0:
                    self.alpha[i] += 1
                    self.alpha_steps = np.row_stack((self.alpha_steps, self.alpha))
            
            if not error:
                break
            
        return alpha
        
    def _gaussian_kernel(self, X, s):
        pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
        return scipy.exp(-pairwise_sq_dists / (2*s**2))
    
    def _polynomial_kernel(self, X, d):
        return (np.matmul(X, np.transpose(X)) + 1) ** d