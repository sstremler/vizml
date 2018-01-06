#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:34:57 2018

@author: stremler
"""

import numpy as np
from vizml.perceptron import KernelPerceptron

np.random.seed(0)

N = 200
X = np.random.normal(loc=0, scale=1, size=(N,2))

# generate classes
# let the class be -1 under the mirror of the hyperbolic cosine function
# and +1 above
y = np.where(X[:,1] <= -1 * np.cosh(X[:,0]) + 1.5, -1, 1)
X[y == 1,1] += 0.5

clf = KernelPerceptron(kernel="polynomial", kernel_param=2)
alpha = clf.fit(X, y)

anim = clf.plot_animation(interval=500, notebook=False, smoothness=400)
print(anim)