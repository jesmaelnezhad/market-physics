#!/bin/python

import requests
import json
import time
import chartslib
import numpy as np
import cmath
from numpy import linalg as LA


charts = []

charts_file = open("charts.index", "r")
charts2 = []
charts_count = int(charts_file.readline())

for i in range(0, charts_count):
    chart = chartslib.Chart(0)
    chart.readFromFile(charts_file)
    charts2.append(chart)

charts_file.close()

print("Period(min)\t# of points")
for chart in charts2:
    print(chart)
print("-----------------------------------")
# clean charts
for chart in charts2:
    #print("Chart size before cleaning: " + str(chart.size()))
    chart.clean()
    #print("Chart size after cleaning: " + str(chart.size()))

new_periods = []
new_intervals = []
i = 0
for chart in charts2:
    if chart.is_usable():
        charts.append(chart)
        new_periods.append(chartslib.periods[i])
        new_intervals.append(chartslib.intervals[i])
    i += 1

chartslib.periods = new_periods
chartslib.intervals = new_intervals


# check charts for consistency
#for chart in charts2:
#    if not chart.check_consistency():
#        print(str(chart.period))
#exit(0)

def all_same_predictions(predictions):
    if len(predictions) == 0:
        return True
    chk = predictions[0][0]
    for p,_ in predictions:
        if chk != p:
            return False
    return True

def printMatrix(M):
    v, W, v_1 = chartslib.decomposeMatrix(M)
    for rIndex in range(0,len(M)):
        row = M[rIndex]
        for i,cell in enumerate(row):
            if i:
                print("\t".expandtabs(3),end='')
            print("{0:0.4f}".format(cell), end='')
        print("| ", end='')
#        row = v[rIndex]
#        for i,cell in enumerate(row):
#            if i:
#                print("\t".expandtabs(3),end='')
#            if isinstance(cell, complex):
#                print("{0:0.4f}".format(cell.real) + "*", end='')
#            else:
#                print("{0:0.4f}".format(cell), end='')
#        print("| ", end='')
        row = W[rIndex]
        for i,cell in enumerate(row):
            if i:
                print("\t".expandtabs(3),end='')
            if isinstance(cell, complex):
                print("{0:0.4f}".format(cell.real) + "*", end='')
            else:
                print("{0:0.4f}".format(cell), end='')
#        print("| ", end='')
#        row = v_1[rIndex]
#        for i,cell in enumerate(row):
#            if i:
#                print("\t".expandtabs(3),end='')
#            if isinstance(cell, complex):
#                print("{0:0.4f}".format(cell.real) + "*", end='')
#            else:
#                print("{0:0.4f}".format(cell), end='')
        print('')

def printJuice(M):
    v, W, v_1 = chartslib.decomposeMatrix(M)
    for cIndex in range(0,len(v)):
        if cIndex:
            print("\t",end='')
        print("{0:0.4f}".format(v[cIndex][0].real), end='')
    print("\t|\t", end='')
    print("{0:0.4f}".format(W[0][0].real), end='')
    print("\t|\t", end='')
    print("{0:0.4f}".format(W[1][1].real), end='')
    print("\t|\t", end='')
    print("{0:0.4f}".format(W[2][2].real), end='')
    print("")

N = len(chartslib.pairs)


#chart = charts[4]
#chart = chart.subChart(["USDCHF","EURUSD", "EURCHF", "EURAUD", "AUDJPY", "EURCAD", "CADJPY"])

#pairCombinations = [
#        ["EURUSD", "USDCHF", "EURCHF"],
#        ["USDCHF","EURUSD", "EURCHF", "EURAUD", "AUDJPY", "EURCAD", "CADJPY"],
#        ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD"] ]

#for comb in pairCombinations:
#    for chart in charts:
#        chart = chart.subChart(comb)
#        M = chart.lastNValuesMatrix(len(comb))
#        printJuice(M)

#exit(0)

#chart.predict_next()



headers = ["Name"] + [str(p) + " Min" for p in chartslib.periods] + ["Considerations"]
print("\t\t".join(headers))
print("----------------------------------------------------------------------------------------------------------------")
for pair in chartslib.pairs:
    pair_predictions = []
    pair_predictions_regression = []
    for chart in charts:
        pair_predictions.append(( chart.predict_next_binary()[pair], chart.gain()[pair] ))
        pair_predictions_regression.append((chart.predict_next_binary_regression()[pair], None))
    print(str(pair) + "\t\t" + "\t\t".join([("B" if p[0] == True else "S") for p in pair_predictions]) + ("\t\t*" if all_same_predictions(pair_predictions) else ""))
    print("      " + "\t\t" + "\t\t".join([("B" if p[0] == True else "S") for p in pair_predictions_regression]) + ("\t\t*" if all_same_predictions(pair_predictions_regression) else ""))
    print("----------------------------------------------------------------------------------------------------------------")
