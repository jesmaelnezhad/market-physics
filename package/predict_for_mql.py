#!/bin/python

import requests
import json
import time
import chartslib
import numpy as np
import cmath
from numpy import linalg as LA


def prepare_chart(data):
    #################################################### Prepare empty results map
    results = {}
    for pair, pair_data in data.items():
        results[pair] = True
    if len(results) == 0:
        return None
    #################################################### Prepare chart
    chart = chartslib.Chart(1) # always using period = 1
    # find the length of the shortest pair data list
    shortest_length = 100000
    for pair, pair_data in data.items():
        if shortest_length > len(pair_data):
            shortest_length = len(pair_data)
    # iterate on data points for all pairs and make points
    for i in range(0, shortest_length):
        # prepare point info that has all pair values
        new_point_info = chartslib.PointInfo()
        for pair, pair_data in data.items():
            new_point_info.addPair(pair, chartslib.PairInfo(pair_data[i], pair_data[i], pair_data[i]))
        # add the new point to the chart
        chart.addPoint(i, new_point_info)
    return chart

def predict_for_mql(data):
    chart = prepare_chart(data)
    if chart == None:
        return None
    return chart.predict_next_binary()

def predict_for_mql_with_regression(data):
    chart = prepare_chart(data)
    if chart == None:
        return None
    pred_ours = chart.predict_next_binary()
    pred_regress = chart.predict_next_binary_regression()
    results = {}
    for pair, pred in pred_ours.items():
        results[pair] = (pred_ours[pair], pred_regress[pair])
    return results

def predict_for_mql_mock(data):
    # Return true for all pairs if we have even number of pairs in total, and false otherwise
    results = {}
    for pair, pair_data in data.items():
        results[pair] = len(data) % 2 == 0
    return results

def predict_for_mql_with_regression_mock(data):
    # Return true for all pairs if we have even number of pairs in total, and false otherwise
    results = {}
    for pair, pair_data in data.items():
        results[pair] = (len(data) % 2 == 0, len(data) % 2 == 0)
    return results

#test_data = {
#       "P1":[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13],
#        "P2":[1,2,3,3,5,6,7,8,9,10,11,12,13],
#        "P4":[1,3,3,4,5,6,7,8,9,10,11,12,13],
#        "P6":[1,2,3,4,5,6,4,8,9,10,11,12,13]}
#
#print(str(predict_for_mql_with_regression(test_data)))
#
