#!/bin/python

import numpy as np
from numpy import linalg as LA

A = np.array([[1,2,3], [3,2,1], [1,0,-1]])
w, v = LA.eig(A)
print("------------------")
print(w)
print("------------------")
print(v)
W = np.diag(w)
v_1 = LA.inv(v)
print("-------")
print(np.matmul(np.matmul(v,W), v_1))
print(np.matmul(A,A))
