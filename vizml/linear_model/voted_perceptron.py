#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:08:42 2018

@author: stremler
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class VotedPerceptron:
    """VotedPerceptron"""
    def __init__(self, eta, epoch):
        self.eta = eta
        self.epoch = epoch
    
    def fit(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.weight_steps = np.ones(self.X.shape[1]).reshape(1,self.X.shape[1])
        self.c_steps = np.array([0])
        self.y = y
        k = 0
        
        while self.epoch > 0:
            for i in range(X.shape[0]):
                if np.dot(self.weight_steps[k,:], self.X[i,]) * y[i] <= 0:
                    self.weight_steps = np.row_stack((self.weight_steps,
                                        self.weight_steps[k,:] + y[i] * self.eta * self.X[i,]))
                    self.c_steps = np.append(self.c_steps, 1)
                    k += 1
                else:
                    self.c_steps[k] += 1
            self.epoch -= 1
        
        return self.weight_steps, self.c_steps
    
    def _predict(self, X, w):
        return np.sign(np.sum(self.c_steps[0:w.shape[0]] * np.sign(np.matmul(X, w.T)), axis=1))
    
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return self._predict(X, self.weight_steps)
    
    def score(self):
        y = self._predict(self.X, self.weight_steps)
        unique, counts = np.unique(-1 * y * self.y, return_counts=True)
        return(counts[0] / np.sum(counts))
        
    def _line_func(self, X, w):
        return np.divide((-w[0] - X * w[1]), w[2], where=w[2]!=0)
    
    def plot_animation(self, interval=400, xinch=5.5, yinch=5, notebook=True, smoothness=400):        
        xlim = (np.min(self.X[:,1]) - 0.2 * np.std(self.X[:,1]), np.max(self.X[:,1]) - 0.2 *  - np.std(self.X[:,1]))
        ylim = (np.min(self.X[:,2]) - 0.2 * np.std(self.X[:,2]), np.max(self.X[:,2]) - 0.2 *  - np.std(self.X[:,2]))
        ycolor = np.where(self.y>0, "#DC0026", "#457CB6")
        yshape = np.where(self.y>0, "o", "^")
        
        xlin = np.linspace(xlim[0], xlim[1], smoothness)
        ylin = np.linspace(ylim[0], ylim[1], smoothness)
        
        xv, yv = np.meshgrid(xlin, ylin)
        pinp = np.c_[xv.ravel(), yv.ravel()]
        pinp = np.column_stack((np.ones(pinp.shape[0]), pinp))
        
        if notebook:
            plt.ioff()
            
        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, 
                             xlim=xlim, ylim=ylim)

        for s, c, x, y in zip(yshape, ycolor, self.X[:,1], self.X[:,2]):
            ax.scatter(x, y, c=c, marker=s)

        conts = []
        for i in range(self.weight_steps.shape[0]):
            z = self._predict(pinp, self.weight_steps[0:i+1,])
            z = z.reshape(xv.shape)
            cont = ax.contourf(xv, yv, z, colors=["#DC0026", "#457CB6"], alpha=1/5)
            te = ax.text(0.10, 0.95, u"Step: {}".format(i), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                         transform=ax.transAxes, ha="center")
            conts.append(cont.collections + [te])

        fig.set_size_inches(xinch, yinch, True)
        anim = animation.ArtistAnimation(fig, conts, interval=interval, blit=False)
        
        return anim