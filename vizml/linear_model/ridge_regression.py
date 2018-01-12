#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 02:49:35 2018

@author: stremler
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from ..preprocessing import PolynomialFeatures, StandardScaler

class RidgeRegression:
    """RidgeRegression"""
    def __init__(self, eta=0.01, maxit=5, alpha=0.5, scale=True, degree=2):
        self.eta = eta
        self.maxit = maxit
        self.alpha = alpha
        self.scale = scale
        self.degree = degree
    
    def fit(self, X, y):
        self.X = X.astype(np.float64)
        self.poly = PolynomialFeatures(degree=self.degree)
        self.poly, self.X = self.poly.fit_transform(self.X)
        self.scaler = StandardScaler()
        self.scaler, self.X[:,1:] = self.scaler.fit_transform(self.X[:,1:])
        self.weight = np.zeros(self.X.shape[1])
        self.weight_steps = np.array([self.weight])
        self.y = y
#        print(self.X)
        for i in range(self.maxit):
            self.weight += self.eta * (np.matmul(self.y - np.matmul(self.X, self.weight), self.X) + self.alpha * self.weight)
            self.weight_steps = np.row_stack((self.weight_steps, self.weight))
            
        return self.weight
    
    def _predict(self, X, w):        
        return self._line_func(X, w)
    
    def predict(self, X):
#        X = np.column_stack((np.ones(X.shape[0]), X))
        X = self.poly.transform(X)
        X[:,1:] = self.scaler.transform(self.X[:,1:])
        return self._predict(X, self.weight)
    
    def score_cost(self):
        return self._score_cost(self.weight)
    
    def _score_cost(self, w):
        y = self._predict(self.X, w)
        return 0.5 * np.sum((self.y - y)**2, axis=1)
        
    def _line_func(self, X, w):
        return np.matmul(w, X.T)
    
    def _init_animation(self):
        self.line.set_data([], [])
        return self.line,

    def _animate(self, i):
        xlim = (np.min(self.X[:,1]), np.max(self.X[:,1]))
        xlin = np.linspace(xlim[0], xlim[1], 100).reshape(100,1)
        
        lineX = self.poly.transform(xlin)        
#        lineX[:,1:] = self.scaler.transform(lineX[:,1:])
        
        lineY = self._predict(lineX, self.weight_steps[i,:])
        self.line.set_data(lineX[:,1], lineY)
        self.title.set_text(u"Step: {}".format(i))
        
        return self.line, self.title
    
    def plot_animation(self, interval=400, xinch=5.5, yinch=5, notebook=True):
        xlim = (np.min(self.X[:,1]) - 0.2 * np.std(self.X[:,1]), np.max(self.X[:,1]) - 0.2 *  - np.std(self.X[:,1]))
        ylim = (np.min(self.y) - 0.2 * np.std(self.y), np.max(self.y) - 0.2 *  - np.std(self.y))
                          
        if notebook:
            plt.ioff()

        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, 
                             xlim=xlim, ylim=ylim)
        plt.axis('on')
        ax.scatter(self.X[:,1], self.y)
        self.line, = ax.plot([], [], c='black')
        self.title = ax.text(0.10, 0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

        fig.set_size_inches(xinch, yinch, True)
        anim = animation.FuncAnimation(fig, func=self._animate, init_func=self._init_animation,
                               frames=self.weight_steps.shape[0], interval=interval, blit=True)
        
        return anim