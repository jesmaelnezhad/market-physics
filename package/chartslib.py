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

# Decomposes a matrix into its eigen vectors
# Returns three matrices v, w, v1 where each row of v is an eigen vector, w is a diagonal matrix of eigen values, and v1 is v's inverse.
# Based on the definition: M = v * w * v1
def decomposeMatrix(M):
    MA = np.array(M)
    w, v = LA.eig(MA)
    W = np.diag(w)
    v_1 = LA.inv(v)
    return v, W, v_1

# Returns the indexth eigen vector/value of the M
def calcEigen(M, index):
    v, w, v_1 = decomposeMatrix(M)
    vec = []
    for i in range(0,len(v)):
        vec.append(v[i][index])
    return vec, w[index][index]

# Uses linear regression to predict the next value of the given sequence
def predictNextLinear(sequence):
    x = np.array([i for i in range(0, len(sequence))]).reshape((-1, 1))
    y = np.array(sequence)
    model = LinearRegression().fit(x, y)
    return model.predict([[-1]])[0]

# The class which represents a pair (e.g., EURUSD is a pair)
# Mostly only value is used in the code.
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

# A point info object, is basically a map from pair to pair info in a certain point of time
# For example:
# pointInfoObj.pairs["EURUSD"] => a PairInfo containing information of for EURUSD
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

# A point is a (x,y) pair, where x is time (or index in sequence)
# and y is a PointInfo object
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

# A Chart object, contains a sequence of points for a certain sampling period.
# For example, for a 5minute interval, we can aggregate all the information of all the pairs
# for the past our as 12 points in a chart.
class Chart:
    def __init__(self, sampling_period):
        self.period = sampling_period
        self.points = []
    def check_consistency(self):
        '''
        Makes sure every point in the chart has values for all pairs and is not incomplete
        '''
        for point in self.points:
            if len(point.y.pairs) != PAIRS_NUMBER:
                return False
        return True
    def clean(self):
        '''
        Removes any point that does not have the value of all pairs
        '''
        new_points = []
        for p in self.points:
            if len(p.y.pairs) == PAIRS_NUMBER:
                new_points.append(p)
        self.points = new_points

    def subChart(self, pairs):
        '''
        Returns another chart object for the given subset of pairs from this chart
        '''
        result = Chart(self.period)
        for point in self.points:
            seq_num = point.x / (-1 * self.period)
            pointValue = PointInfo()
            for pair,pairInfo in point.y.pairs.items():
                if pair in pairs:
                    pointValue.addPair(pair, pairInfo)
            result.addPoint(seq_num, pointValue)
        return result
    def is_usable(self):
        '''
        A chart is usable only if it has at least two points
        '''
        return len(self.points) >= 2
    def size(self):
        '''
        Returns the number of points in the chart
        '''
        return len(self.points)
    def __str__(self):
        return str(self.period) + "\t\t" + str(len(self.points))
    def addPoint(self, seq_number, pointValue):
        '''
        Adds a new point to the chart based on the given sequence number and value
        '''
        point = Point(-1 * seq_number * self.period, pointValue)
        self.points.append(point)
    def predictEigen(self):
        '''
        Suppose we have 5 pairs in each point of the chart. The latest 5 points can be used
        to form a 5x5 matrix. Call this matrix A0. Now suppose we create a similar matrix
        except we ignore the latest K points, called AK.
        Let's call the first eigen vector and the first eigen value of Ai, V1_i, and w1_i.
        So for the matrix sequence of (A7, A6, A5, A4, A3, A2, A1, A0) we have the corresponding
        vector sequence of first eigen vectors (V1_7, V1_6, V1_5, ..., V1_0) and the corresponding 
        value sequence of first eigen values (w1_7, ..., w1_0).

        If a new point arrive, we can form a new matrix (called A-1).

        This function predicts the first eigen vector and value of A-1, (V1_-1, w1_-1), based on
        the sequence (V1_0, w1_0), (V1_1, w1_1), .... from the past in the chart.
        '''
        N = self.getNumberOfPairs()
        # calculate eigen values and vectors
        eigenValues = []
        eigenVectors = []
        for pairIndex in range(0, N):
            eigenVectors.append([])

        # Why do we move the window N times if we use linear regression?
        # Has no reason. The N used in the next line can be changed to any number.
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
        '''
        Returns a N * X matrix with values of each point of time in a column.
        X will be the number of pairs the chart.
        Column zero of the retuening matrix would be the newest point
        '''
        return self.lastNValuesMatrixWithOffset(N, 0)
    def lastNValuesMatrixWithOffset(self, N, offset):
        '''
        Just like lastNValuesMatrix, only ignores the newst offset points.
        '''
        result = []
        for pair in self.points[0].y.pairs:
            pairResult = []
            for i in range(0,N):
                pairResult.append(self.points[i+offset].y.pairs[pair].value)
            result.append(pairResult)
        return result
    def getLastPoint(self):
        '''
        Returns the newst (the latest) point.
        '''
        return self.points[0]
    def getNumberOfPairs(self):
        '''
        Returns the number of points in the chart
        '''
        return len(self.getLastPoint().y.pairs)
    def getPairs(self):
        '''
        Returns all pairs of the latest point.
        Basically, returns a list of pairs that are expected to be found in each point (if chart was cleaned).
        '''
        pairs = []
        for p,_ in self.getLastPoint().y.pairs.items():
            pairs.append(p)
        return pairs
    def calculate_gain(self, new_value, old_value):
        '''
        Utility function to return gain/loss in percentages based on the given new and old values
        '''
        if new_value > old_value:
            return (new_value - old_value) * 100.0 / old_value
        else:
            return (old_value - new_value) * 100.0 / old_value
    def predict_next(self):
        '''
        Returns a dictionary from pairs to the next predicted value for each pair
        Uses our method to predict the next point that is expected to be added to the chart (main prediction method).
        '''
        # predict the next first eigen value/vector
        # vec => the predicted first eigen vector of the NxN matrix whose first column is the future point
        # w   => the predicted first eigen value of the NxN matrix whose first column is the future point
        vec, w = self.predictEigen()
        # Get the last N-1 columns from the chart to form a N-1*N matrix to be used to form the equation
        # [nextPoint column of Xs to be found | N-1 columns of the last N-1 points] should = to predicted vector * predicted value * inverse of predicted vector
        # So
        # New_NxN_Matrix * Predicted_V0 = Predicted_V0 * Predicted_W0
        #
        # So if you simplify the equation, the next value for the ith pair should be:
        # x_i = ( w * vec[i] - (M[i][:] . vec[1:N-1]) ) / vec[0]
        # 
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
    def gain(self):
        '''
        Returns a map from pairs to predicted gain/loss percentage values
        Calculates the predicted gain if we use our prediction method for predicting the next value for each pair
        ''' 
        prediction = self.predict_next()
        pair_gains = {}
        for pair, pair_prediction in prediction.items():
            new_value = pair_prediction
            old_value = self.getLastPoint().y.pairs[pair].value
            pair_gains[pair] = self.calculate_gain(new_value, old_value)
        return pair_gains
    def predict_next_regression(self):
        '''
        Predicts the next value for each pair (and returns a dictionary just like predict_next) using linear regression
        '''
        M = self.lastNValuesMatrix(self.getNumberOfPairs())
        pairs = self.getPairs()
        result = {}
        for i, pair in enumerate(pairs):
            result[pair] = predictNextLinear(M[i])
        return result
    def predict_next_binary(self):
        '''
        Uses our method for perdiction
        Returns a map from pair to True/False based on whether or not the prediction is increasing or descreasing relative to latest point
        '''
        return self.convert_prediction_to_binary(self.predict_next())
    def predict_next_binary_regression(self):
        '''
        Uses our linear regression for perdiction
        Returns a map from pair to True/False based on whether or not the prediction is increasing or descreasing relative to latest point
        '''
        return self.convert_prediction_to_binary(self.predict_next_regression())
    def convert_prediction_to_binary(self, predictions):
        '''
        Helper method to convert prediction to True or False (Higher than now or lower than now)
        '''
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
    def writeToFile(self, f):
        '''
        Write chart to given file
        '''
        f.write(str(self.period) + '\n')
        f.write(str(len(self.points)) + '\n')
        for point in self.points:
            point.writeToFile(f)
    def readFromFile(self, f):
        '''
        Read chart from the given file
        '''
        self.period = int(f.readline())
        points_count = int(f.readline())
        for i in range(0, points_count):
            new_point = Point(0, 0)
            new_point.readFromFile(f)
            self.points.append(new_point)
