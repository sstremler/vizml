#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 02:51:36 2018

@author: stremler
"""

import numpy as np
from vizml.linear_model import PolynomialRegression

np.random.seed(0)

X = np.linspace(-4, 2.5, 100).reshape(100,1)
y = X**3/4 + 3*X**2/4 - 3*X/2 + np.random.normal(loc=0, scale=0.2, size=(100,1))
y = y.ravel()

clf = PolynomialRegression(eta=0.001, maxit=500, alpha=0, scale=True, degree=3)
w = clf.fit(X, y)
print(clf.weight_steps)

anim = clf.plot_animation(interval=20, notebook=False, xinch=5.5, yinch=5)
print(anim)

#anim_traj = clf.plot_trajectory(interval=400, notebook=False, xinch=5.5, yinch=5)
#print(anim_traj)