#!/bin/python

import requests
import json
import time
import chartslib
import numpy as np
import cmath
from numpy import linalg as LA

charts = chartslib.prepare_charts_to_use("charts.index")
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

#N = len(chartslib.pairs)


#chart = charts[4]
#chart = chart.subChart(["USDCHF","EURUSD", "EURCHF", "EURAUD", "AUDJPY", "EURCAD", "CADJPY"])

#for chart in charts:
#    for pair in chart.getPairs():
#        print(pair)
#        values = chart.allPairValues(pair)
#        print(values)

#exit(0)

#chart.predict_next()


def print_chart_set(charts):
    print("Chart covariance matrices:")
    for chart in charts:
        print("Chart " + str(chart.period))
        #cov = chart.getCorrelationCoefficient()
        chart.modularize()
        #for i in range(0,23):
        #    cov = np.cov(cov)
        #print(cov.tolist())
    
    tabs = "\t\t"
    headers = ["Name"] + [str(p) + " Min" for p in chartslib.periods] + ["Considerations"]
    firstEigenValues = ["E."] + [str(chart.predict_first_eigen_value()) + "/" + str(int(chart.predict_first_eigen_value()/len(chart.get_pairs())))  for chart in charts]
    print(tabs.join(headers))
    print(tabs.join(firstEigenValues))
    print("----------------------------------------------------------------------------------------------------------------")
    for pair in charts[0].get_pairs():
        pair_predictions = []
        pair_predictions_regression = []
        pair_atrs = []
        pair_predictions_changes = []
        for chart in charts:
            pair_predictions.append(( chart.predict_next_binary()[pair], None ))
            pair_predictions_regression.append((chart.predict_next_binary_regression()[pair], None))
            pair_atrs.append((chart.average_recent_range()[pair], None))
            pair_predictions_changes.append((chart.predict_next_change_range()[pair], None))
        print(str(pair) + tabs + tabs.join([("B" if p[0] == True else "S") for i,p in enumerate(pair_predictions)]) + ("\t\t*" if all_same_predictions(pair_predictions) else ""))
        print("      " + tabs + tabs.join([(str(p[0])) for p in pair_predictions_changes]))
        print("      " + tabs + tabs.join([(str(p[0])) for p in pair_atrs]))
        print("      " + tabs + tabs.join([("B" if p[0] == True else "S") for p in pair_predictions_regression]) + ("\t\t*" if all_same_predictions(pair_predictions_regression) else ""))
        print("----------------------------------------------------------------------------------------------------------------")

def print_chart_summary(chart):
    M = chart.matrix(5)
    for i,row in enumerate(M):
        print(chart.get_last_point().y.get_pair_at(i) + "\t",end='')
        for p in row:
            print(str(p) + "\t",end='')
        print()
#for chart in charts:
#    chart.decompose()
#    #print_chart_summary(chart)
#    #curr_chart = chart.materialize_by_currency("USD")
#    #print_chart_summary(curr_chart)


#printChartSet(charts)

def check_combinations():
    print("----------------------------------------------------------------------------------------------------------------")
    print("Checking different combinations and orders ...")
    for i, pairCombination in enumerate(chartslib.pairCombinations):
        newCharts = []
        for chart in charts:
            newCharts.append(chart.sub_chart(pairCombination))
        print("Combination is : " + "\t".join([p for p in pairCombination]))
        print_chart_set(newCharts)


#for chart in charts:
#    print("Modules for chart " + str(chart.period) + "min")
#    modules = chart.modularize()
#    for module in modules:
#        print(str(len(module)) + "\t", end='')
#    print()
#chartmodules = []
#for chart in charts:
#    modules = chart.modularize()
#    for module in modules:
#        newCharts = []
#        for chart in charts:
#            newCharts.append(chart.sub_chart(module))
#        print("Module is : " + "\t".join([p for p in module]))
#        print_chart_set(newCharts)
#
#check_combinations()

