#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 00:55:52 2018

@author: stremler
"""

import numpy as np
import scipy
from scipy.spatial.distance import cdist
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
                    error = True
            
            if not error:
                break
            
        return self.alpha
        
    def _gaussian_kernel(self, X, s):
        pairwise_sq_dists = cdist(self.X, X, 'sqeuclidean')
        return scipy.exp(-pairwise_sq_dists / (2*s**2))
    
    def _polynomial_kernel(self, X, d):
        return (np.matmul(self.X, np.transpose(X)) + 1) ** d
        
    def _predict(self, X, alpha):        
        return np.sign(np.matmul(alpha * self.y, self.kernel(X, self.kernel_param)))
    
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return self._predict(X, self.alpha)
    
    def score(self):
        y = self._predict(self.X)
        unique, counts = np.unique(-1 * y * self.y, return_counts=True)
        return(counts[0] / np.sum(counts))
    
    def plot_animation(self, interval=400, xinch=5, yinch=5, notebook=True, smoothness=400):        
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
        for i in range(self.alpha_steps.shape[0]):
            z = self._predict(pinp, self.alpha_steps[i,])
            z = z.reshape(xv.shape)
            cont = ax.contourf(xv, yv, z, colors=["#DC0026", "#457CB6"], alpha=1/5)
            te = ax.text(0.10, 0.95, u"Step: {}".format(i), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                         transform=ax.transAxes, ha="center")
            conts.append(cont.collections + [te])

        fig.set_size_inches(xinch, yinch, True)
        anim = animation.ArtistAnimation(fig, conts, interval=interval, blit=False)
        
        return anim