#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:13:50 2018

@author: stremler
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class LinearRegression:
    """LinearRegression"""
    def __init__(self, eta=0.01, maxit=5):
        self.eta = eta
        self.maxit = maxit
    
    def fit(self, X, y):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.weight = np.zeros(self.X.shape[1])
        self.weight_steps = np.array([self.weight])
        self.y = y
        
        for i in range(self.maxit):
            self.weight += np.matmul(self.eta * (self.y - np.matmul(self.X, self.weight)), self.X)
            self.weight_steps = np.row_stack((self.weight_steps, self.weight))
            
        return self.weight
    
    def _predict(self, X, w):        
        return self._line_func(X, w)
    
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
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
        lineX = np.array([np.min(self.X[:,1]), np.max(self.X[:,1])])
        lineX = np.column_stack((np.ones(lineX.shape[0]), lineX))
        lineY = self._predict(lineX, self.weight_steps[i,:])
        self.line.set_data(lineX[:,1:], lineY)
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
        
    def plot_trajectory(self, interval=400, xinch=5.5, yinch=5, notebook=True, smoothness=100):
        blim = (np.min(self.weight_steps[:,0]) - 0.2 * np.std(self.weight_steps[:,0]), np.max(self.weight_steps[:,0]) - 0.2 *  - np.std(self.weight_steps[:,0]))
        xlim = (np.min(self.weight_steps[:,1]) - 0.2 * np.std(self.weight_steps[:,1]), np.max(self.weight_steps[:,1]) - 0.2 *  - np.std(self.weight_steps[:,1]))
        
        blin = np.linspace(blim[0], blim[1], smoothness)
        xlin = np.linspace(xlim[0], xlim[1], smoothness)
        
        xv, yv = np.meshgrid(blin, xlin)
        weights = np.c_[xv.ravel(), yv.ravel()]
        
        if notebook:
            plt.ioff()

        fig = plt.figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, autoscale_on=False, 
                             xlim=blim, ylim=xlim)
        plt.axis('on')
        error = self._score_cost(weights).reshape(smoothness, smoothness)
        ax.contour(blin, xlin, error)

        conts = []
        for i in range(self.weight_steps.shape[0]-1):
            add_arts = []
            
            for j in range(i+1):
                b = self.weight_steps[j,0]
                x = self.weight_steps[j,1]
                db = self.weight_steps[j+1,0] - b
                dx = self.weight_steps[j+1,1] - x
                
                cont = ax.arrow(b, x, db, dx, color="red")
                add_arts.append(cont)
                
            te = ax.text(0.10, 0.95, u"Step: {}".format(i), bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                         transform=ax.transAxes, ha="center")
            conts.append(add_arts + [te])

        fig.set_size_inches(xinch, yinch, True)
        anim = animation.ArtistAnimation(fig, conts, interval=interval, blit=False)
        
        return anim