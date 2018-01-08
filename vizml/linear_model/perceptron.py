#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 01:19:11 2018

@author: stremler
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Perceptron:
    """Perceptron"""
    def __init__(self, eta):
        self.eta = eta
    
    def fit(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.weight = np.ones(self.X.shape[1])
        self.weight_steps = self.weight
        self.y = y
        
        while True:
            error = 0
            
            for i in range(X.shape[0]):
                if np.dot(self.weight, self.X[i,]) * y[i] <= 0:
                    self.weight = self.weight + y[i] * self.eta * self.X[i,]
                    self.weight_steps = np.row_stack((self.weight_steps, self.weight))
                    error += 1
            
            if error == 0:
                break
            
        return self.weight
    
    def _predict(self, X):        
        return np.sign(np.sum(X * self.weight, axis=1))
    
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return self._predict(X)
    
    def score(self):
        y = self._predict(self.X)
        unique, counts = np.unique(-1 * y * self.y, return_counts=True)
        return(counts[0] / np.sum(counts))
        
    def _line_func(self, X, w):
        return np.divide((-w[0] - X * w[1]), w[2], where=w[2]!=0)
    
    def _init_animation(self):
        self.line.set_data([], [])
        return self.line,

    def _animate(self, i):
        lineX = np.array([np.min(self.X[:,1]), np.max(self.X[:,1])])
        lineY = self._line_func(lineX, self.weight_steps[i,])
        self.line.set_data(lineX, lineY)
        self.title.set_text(u"Step: {}".format(i))
        
        return self.line, self.title
    
    def plot_animation(self, interval=400, xinch=5, yinch=5, notebook=True):        
        xlim = (np.min(self.X[:,1]) - 0.2 * np.std(self.X[:,1]), np.max(self.X[:,1]) - 0.2 *  - np.std(self.X[:,1]))
        ylim = (np.min(self.X[:,2]) - 0.2 * np.std(self.X[:,2]), np.max(self.X[:,2]) - 0.2 *  - np.std(self.X[:,2]))
        ycolor = np.where(self.y>0, "#DC0026", "#457CB6")
        yshape = np.where(self.y>0, "o", "^")
                          
        if notebook:
            plt.ioff()
            
        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, 
                             xlim=xlim, ylim=ylim)
        plt.axis('on')
        for s, c, x, y in zip(yshape, ycolor, self.X[:,1], self.X[:,2]):
            ax.scatter(x, y, c=c, marker=s)
        self.line, = ax.plot([], [], c='black')
        self.title = ax.text(0.10, 0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

        fig.set_size_inches(xinch, yinch, True)
        anim = animation.FuncAnimation(fig, func=self._animate, init_func=self._init_animation,
                               frames=self.weight_steps.shape[0], interval=interval, blit=True)
        
        return anim