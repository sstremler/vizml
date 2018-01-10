#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:04:40 2018

@author: stremler
"""

import numpy as np
from vizml.linear_model import LogisticRegression

np.random.seed(0)

N = 100
X = np.random.normal(loc=1.2, scale=1, size=(N,2))
X = np.row_stack((X, np.random.normal(loc=0, scale=1, size=(N,2))))
y = np.row_stack((np.full((N,1), -1), np.full((N,1), 1))).ravel()

clf = LogisticRegression(eta=0.001, maxit=10)
w = clf.fit(X, y)

anim = clf.plot_animation(interval=400, notebook=False)
print(anim)