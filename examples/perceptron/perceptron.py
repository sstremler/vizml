#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:51:50 2018

@author: stremler
"""

import numpy as np
from vizml.perceptron import Perceptron

np.random.seed(0)

N = 10
X = np.random.normal(loc=0, scale=1, size=(N,2))
X = np.row_stack((X, np.random.normal(loc=3.5, scale=1, size=(N,2))))
y = np.row_stack((np.full((N,1), -1), np.full((N,1), 1))).ravel()

clf = Perceptron(eta=0.005)
w = clf.fit(X, y)

anim = clf.plot_animation(interval=20, notebook=False)
print(anim)