#!/bin/python

import chartslib

def printMatrix(M):
    v, W, v_1 = chartslib.decomposeMatrix(M)
    for rIndex in range(0,len(M)):
        row = M[rIndex]
        for i,cell in enumerate(row):
            if i:
                print("\t".expandtabs(3),end='')
            print("{0:0.4f}".format(cell.real), end='')
        print()

r1 = [1, 1, 1]
r2 = [2, 2, 2]
r3 = [3, 3, 3]

M1 = [r1, r2, r3]
#M2 = [r2, r1, r3]

v1, w1, _ = chartslib.decomposeMatrix(M1)
printMatrix(v1)
print(w1)
print("=======================================")
#v2, w2, _ = chartslib.decomposeMatrix(M2)
#printMatrix(v2)
#print(w2)
M1[0] = [2,2,2]
v1, w1, _ = chartslib.decomposeMatrix(M1)
printMatrix(v1)
print(w1)
print("=======================================")
M1[0] = [1,1,1]
M1[1] = [4,4,4]
v1, w1, _ = chartslib.decomposeMatrix(M1)
printMatrix(v1)
print(w1)
print("=======================================")
M1[1] = [2,2,2]
M1[2] = [6,6,6]
v1, w1, _ = chartslib.decomposeMatrix(M1)
printMatrix(v1)
print(w1)
print("=======================================")
