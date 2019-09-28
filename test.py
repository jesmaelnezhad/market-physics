import requests
import json
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import cmath
from numpy import linalg as LA
from datetime import datetime


def decompose_matrix(M, factor=1):
    '''
    Decompose the given matrix using eigen values and eigen vectors.
    '''
    MA = np.array(M)
    w, v = LA.eig(MA)
    W = np.diag(w)
    #W[0][0] = factor * W[0][0]
    for i in range(0,len(M)):
        v[i][0] *= factor
    v_1 = LA.inv(v)
    return v, W, v_1

def combine(v, w, v_1):
    return np.matmul(np.matmul(v,w),v_1)


def res():
    return (1,2)

(a,b) = res()
print(str(a) + str(b))

print(np.matmul(np.array([[1,2]]), np.array([[1],[2]])))

M = [[1,2,3],[4,5,6],[7,8,9]]
v,w,v_1=decompose_matrix(M)
print(v,w,v_1)
print(combine(v,w,v_1))
print("-----------")
v,w,v_1=decompose_matrix(M, factor=-1)
print(v,w,v_1)
print(combine(v,w,v_1))
print("-----------")
