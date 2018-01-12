#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 02:51:36 2018

@author: stremler
"""

import numpy as np
from vizml.linear_model import RidgeRegression
from vizml.preprocessing import PolynomialFeatures, StandardScaler
#import matplotlib.pyplot as plt

np.random.seed(0)

fs = 400 # sample rate
f = 5    # frequency
X = np.array([np.arange(100)]).T
y = np.sin(2 * np.pi * f * X.T / fs) + np.random.normal(loc=0, scale=0.2, size=(100,))
y = y.ravel()
#plt.scatter(X, y)

clf = RidgeRegression(eta=0.0001, maxit=1000, alpha=0, scale=True, degree=20)
w = clf.fit(X, y)
print(clf.weight_steps)

anim = clf.plot_animation(interval=20, notebook=False, xinch=5.5, yinch=5)
print(anim)

#anim_traj = clf.plot_trajectory(interval=400, notebook=False, xinch=5.5, yinch=5)
#print(anim_traj)