#!/bin/python

import requests
import json
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import cmath
from numpy import linalg as LA

periods = [1, 5, 15, 30, 60]
#periods = [60]

intervals = ["1min", "5min", "15min", "30min", "60min"]
#intervals = ["1min"]

pairs = ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD"]
#pairs = ["USDCHF"]

PAIRS_NUMBER = len(pairs)


def decomposeMatrix(M):
    MA = np.array(M)
    w, v = LA.eig(MA)
    W = np.diag(w)
    v_1 = LA.inv(v)
    return v, W, v_1
def calcEigen(M, index):
    v, w, v_1 = decomposeMatrix(M)
    vec = []
    for i in range(0,len(v)):
        vec.append(v[i][index])
    return vec, w[index][index]

def predictNextLinear(sequence):
    x = np.array([i for i in range(0, len(sequence))]).reshape((-1, 1))
    y = np.array(sequence)
    model = LinearRegression().fit(x, y)
    return model.predict([[-1]])[0]


class PairInfo:
    def __init__(self, value, high, low):
        self.value = value
        self.high = high
        self.low = low
    def writeToFile(self, f):
        f.write(str(self.value) + '\n')
        f.write(str(self.high) + '\n')
        f.write(str(self.low) + '\n')
    def readFromFile(self, f):
        self.value = float(f.readline())
        self.high = float(f.readline())
        self.low = float(f.readline())

class PointInfo:
    def __init__(self):
        self.pairs = {}
    def addPair(self, pair, pairInfo):
        self.pairs[pair] = pairInfo
    def writeToFile(self, f):
        f.write(str(len(self.pairs)) + '\n')
        for pair, pairInfo in self.pairs.items():
            f.write(str(pair) + '\n')
            pairInfo.writeToFile(f)
    def readFromFile(self, f):
        pairs_count = int(f.readline())
        for i in range(0, pairs_count):
            pair = f.readline().strip()
            pairInfo = PairInfo(0,0,0)
            pairInfo.readFromFile(f)
            self.pairs[pair] = pairInfo

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "(" + str(x) + ", " + str(y) + ")"
    def writeToFile(self, f):
        f.write(str(self.x) + '\n')
        self.y.writeToFile(f)
    def readFromFile(self, f):
        self.x = int(f.readline())
        new_point_info = PointInfo()
        new_point_info.readFromFile(f)
        self.y = new_point_info


class Chart:
    def __init__(self, sampling_period):
        self.period = sampling_period
        self.points = []
    def check_consistency(self):
        for point in self.points:
            if len(point.y.pairs) != PAIRS_NUMBER:
                return False
        return True
    def subChart(self, pairs):
        result = Chart(self.period)
        for point in self.points:
            seq_num = point.x / (-1 * self.period)
            pointValue = PointInfo()
            for pair,pairInfo in point.y.pairs.items():
                if pair in pairs:
                    pointValue.addPair(pair, pairInfo)
            result.addPoint(seq_num, pointValue)
        return result
    def clean(self):
        new_points = []
        for p in self.points:
            if len(p.y.pairs) == PAIRS_NUMBER:
                new_points.append(p)
        self.points = new_points
    def is_usable(self):
        return len(self.points) >= 2
    def size(self):
        return len(self.points)
    def __str__(self):
        return str(self.period) + "\t\t" + str(len(self.points))
    def addPoint(self, seq_number, pointValue):
        point = Point(-1 * seq_number * self.period, pointValue)
        self.points.append(point)
    def predictEigen(self):
        N = self.getNumberOfPairs()
        # calculate eigen values and vectors
        eigenValues = []
        eigenVectors = []
        for pairIndex in range(0, N):
            eigenVectors.append([])
        for offset in range(0, N):
            M = self.lastNValuesMatrixWithOffset(N, offset)
            vec, w = calcEigen(M, 0)
            # add w to the list to be used for regression
            eigenValues.append(w.real)
            # add vec elements to the list to be used for regression
            for i, vec_element in enumerate(vec):
                eigenVectors[i].append(vec_element.real)
        # predict w
        w = predictNextLinear(eigenValues)
        # predict vector
        vec = []
        for v in eigenVectors:
            v_pred = predictNextLinear(v)
            vec.append(v_pred)
        return vec, w
    def lastNValuesMatrix(self, N):
        return self.lastNValuesMatrixWithOffset(N, 0)
    def lastNValuesMatrixWithOffset(self, N, offset):
        result = []
        for pair in self.points[0].y.pairs:
            pairResult = []
            for i in range(0,N):
                pairResult.append(self.points[i+offset].y.pairs[pair].value)
            result.append(pairResult)
        return result
    def getLastPoint(self):
        return self.points[0]
    def getNumberOfPairs(self):
        return len(self.getLastPoint().y.pairs)
    def getPairs(self):
        pairs = []
        for p,_ in self.getLastPoint().y.pairs.items():
            pairs.append(p)
        return pairs
    def slope(self):
        return self.slope(0)
    def slope(self, start_granularity):
        length_slopes = []
        lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        usable_length = 0
        for i in range(0, len(lengths)):
            if lengths[i] < len(self.points):
                usable_length = i+1
        lengths = lengths[0:usable_length]
        for length in lengths:
            if length >= start_granularity:
                length_slopes.append(self.slope_for(length))
        slope = {}
        #print(usable_length)
        for pair,_ in length_slopes[0].items():
            pair_slopes = []
            for length_index in range(0,len(length_slopes)):
                length_slope = length_slopes[length_index]
                if pair not in length_slope.keys():
                    continue
                #print(length_slope[pair])
                pair_slopes.append(length_slope[pair])
                #print(len(pair_slopes))
            slope[pair] = self.merge_slopes(pair_slopes)
        return slope
    def calc_predicted_y_difference(self, future_sequence_number):
        x_difference = future_sequence_number * self.period
        y_difference = {}
        for pair, slope in self.slope(future_sequence_number).items():
            y_difference[pair] = x_difference * slope
        return y_difference
    def calculate_gain(self, new_value, old_value):
        if new_value > old_value:
            return (new_value - old_value) * 100.0 / old_value
        else:
            return (old_value - new_value) * 100.0 / old_value
    def gain(self):
        prediction = self.predict_next()
        pair_gains = {}
        for pair, pair_prediction in prediction.items():
            new_value = pair_prediction
            old_value = self.getLastPoint().y.pairs[pair].value
            pair_gains[pair] = self.calculate_gain(new_value, old_value)
        return pair_gains
    def predict_next_regression(self):
        M = self.lastNValuesMatrix(self.getNumberOfPairs())
        pairs = self.getPairs()
        result = {}
        for i, pair in enumerate(pairs):
            result[pair] = predictNextLinear(M[i])
        return result
    def predict_next(self):
        vec, w = self.predictEigen()
        M = self.lastNValuesMatrix(self.getNumberOfPairs()-1)
        predictions = []
        for i, seq in enumerate(M):
            dotSum = 0
            for seq_i in range(0, len(seq)):
                dotSum += seq[seq_i] * vec[seq_i+1]
            prediction = (w * vec[i] - dotSum) / vec[0]
            predictions.append(prediction)
        pairs = self.getPairs()
        result = {}
        for i, pair in enumerate(pairs):
            result[pair] = predictions[i]
        return result
    def predict_next_binary(self):
        return self.convert_prediction_to_binary(self.predict_next())
    def predict_next_binary_regression(self):
        return self.convert_prediction_to_binary(self.predict_next_regression())
    def convert_prediction_to_binary(self, predictions):
        C = self.lastNValuesMatrix(1)
        results = {}
        i = 0
        for pair, pred in predictions.items():
            if pred > C[i][0]:
                results[pair] = True
            else:
                results[pair] = False
            i += 1
        return results
    def predict(self, future_sequence_number):
        y_difference = self.calc_predicted_y_difference(future_sequence_number)
        prediction = {}
        for pair, y_diff in y_difference.items():
            prediction[pair] = self.getLastPoint().y.pairs[pair].value + y_diff
        return prediction
    def predict_binary(self, future_sequence_number):
        y_difference = self.calc_predicted_y_difference(future_sequence_number)
        prediction = {}
        for pair, y_diff in y_difference.items():
            if y_diff < 0:
                prediction[pair] = False
            else:
                prediction[pair] = True
        return prediction

    def merge_slopes(self, slopes):
        merge = 0
        for slope in slopes:
            merge += slope
        merge = merge / len(slopes)
        return merge
    def slope_for(self, length):
        if len(self.points) < length:
            return {}
        point1 = self.points[length]
        point2 = self.points[0]
        return self.slope_between(point1, point2)
    def slope_between(self, point1, point2):
        if point1.x == point2.x:
            return {}
        slopes = {}
        for pair,pairInfo in point1.y.pairs.items():
            y_difference = point2.y.pairs[pair].value - pairInfo.value
            x_difference = point2.x - point1.x
            slope = y_difference / x_difference
            slopes[pair] = slope
        return slopes
    def writeToFile(self, f):
        f.write(str(self.period) + '\n')
        f.write(str(len(self.points)) + '\n')
        for point in self.points:
            point.writeToFile(f)
    def readFromFile(self, f):
        self.period = int(f.readline())
        points_count = int(f.readline())
        for i in range(0, points_count):
            new_point = Point(0, 0)
            new_point.readFromFile(f)
            self.points.append(new_point)
