#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:53:26 2018

@author: stremler
"""

import numpy as np
from vizml.linear_model import LinearRegression

np.random.seed(0)

# generate data with r = 0.8 correlation
N = 30
X = np.random.normal(loc=0, scale=1, size=(N,1))
r = 0.8
y = r*X + np.random.normal(loc=0, scale=np.sqrt(1-r**2), size=(N,1))
y = y.ravel()
print(np.corrcoef(X.T,y))

clf = LinearRegression(eta=0.01, maxit=15)
w = clf.fit(X, y)

anim = clf.plot_animation(interval=400, notebook=False, xinch=5.5, yinch=5)
print(anim)

anim_traj = clf.plot_trajectory(interval=400, notebook=False, xinch=5.5, yinch=5)
print(anim_traj)