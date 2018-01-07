#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:05:47 2018

@author: stremler
"""

import numpy as np
from vizml.perceptron import VotedPerceptron

np.random.seed(0)

N = 100
X = np.random.normal(loc=1.2, scale=1, size=(N,2))
X = np.row_stack((X, np.random.normal(loc=0, scale=1, size=(N,2))))
y = np.row_stack((np.full((N,1), -1), np.full((N,1), 1))).ravel()

clf = VotedPerceptron(eta=0.5, epoch=10)
weight_steps, c_steps = clf.fit(X, y)

anim = clf.plot_animation(interval=400, notebook=False, smoothness=400)
print(anim)