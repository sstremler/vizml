#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:57:15 2018

@author: stremler
"""

import numpy as np
import random
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
        self.layers = np.concatenate(([self.X.shape[1] - 1], self.layers, [self.y.shape[1]]))
        w = self._initialize_weights(self.layers)
        self.weight_steps = []
        w_temp = []
        
        for i in range(self.epoch): # iterate over epochs
            np.random.seed(i)
            itrange = list(range(self.X.shape[0]))
            random.shuffle(itrange)
            
            for k in itrange: # iterate over points randomly
                # feedforward
                layer_sum = []
                layer_output = []
                layer_output.append(self.X[k,:])
                
                for j in range(len(self.layers) - 2): # only for hidden layers
#                    print(w[j].shape)
                    layer_h_sum = np.matmul(layer_output[j], w[j])
                    layer_sum.append(layer_h_sum)
                    layer_h_output = self._transfer_f(layer_h_sum)
                    layer_output.append(np.concatenate(([1], layer_h_output)))
                    
                # output layer
                last_layer_output = layer_output[len(layer_output) - 1] # (n-1)st layer
                layer_o_sum = np.matmul(last_layer_output, w[len(w) - 1]) # output layer sum
                layer_o_output = layer_o_sum # linear output for regression
                # end output layer
            
                # calculate error
                layer_o_output_error = self.y[k,:] - layer_o_output
                layer_o_output_error_total = 0.5*np.sum(layer_o_output_error**2) # SSE
                # end calculate error
                # end feedforward
                
                # backpropagation
                # output layer
                last_layer_output_wo_b = np.atleast_2d(last_layer_output[1:]) # output of the (n-1)st layer without bias
                delta_output = np.atleast_2d(-1*layer_o_output_error)
                grad_w_output = np.matmul(delta_output, last_layer_output_wo_b)
                grad_w_output = np.row_stack((delta_output, grad_w_output.T))
                w_temp = []
                w_temp.append(w[len(w) - 1] - self.eta * grad_w_output)
                # end output layer
                
                # hidden layer
#                print(1)
#                print(len(self.layers))
                for j in reversed(range(len(self.layers) - 2)):
                    print(j)
                    delta_h = self._transfer_f_derivative(layer_sum[j]) * \
                                np.matmul(delta_output, w[j + 1][1:,:].T)
                    grad_w_h = np.matmul(delta_h.T, layer_output[j][1:])
                    grad_w_h = np.row_stack((delta_h, grad_w_h))
                    w_temp.insert(0, w[j] - self.eta*grad_w_h)
                    delta_output = delta_h
                # hl end
                # end backpropagation
                print(w_temp)
                print("\n")
                w = w_temp
                
            self.weight_steps.append(w)
            
        return w
                
        
    def _initialize_weights(self, layers):
        w = []
        
        for i in range(len(layers) - 1):
            # number of neurons in the i-th and i+1-th layer
            n_no = (layers[i] + 1) + layers[i + 1]
            
            w.append(np.random.uniform(low=-np.sqrt(6/n_no), high=np.sqrt(6/n_no), 
                              size=(layers[i] + 1, layers[i + 1])))
            
        return w
    
    def _transfer_f(self, x):
        return 1/(1 + np.exp(-x))
    
    def _transfer_f_derivative(self, x):
        return self._transfer_f(x) * (1 - self._transfer_f(x))