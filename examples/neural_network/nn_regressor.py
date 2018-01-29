#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 01:47:57 2018

@author: stremler
"""

import numpy as np
from vizml.neural_network import NNRegressor

np.random.seed(0)

X = np.linspace(-4, 2.5, 100).reshape(100,1)
y = X**3/4 + 3*X**2/4 - 3*X/2 + np.random.normal(loc=0, scale=0.2, size=(100,1))
y = y.ravel()

clf = NNRegressor(eta=0.001, layers=3, epoch=1)
clf.fit(X, y)

#anim = clf.plot_animation(interval=500, notebook=False, smoothness=400)
#print(anim)