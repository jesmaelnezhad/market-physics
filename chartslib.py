#!/bin/python

import requests
import json
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import cmath
from numpy import linalg as LA
from datetime import datetime
from enum import Enum

NORM_FACTOR = 1.0

CORRELATION_PRUNE_FACTOR = 0.9
CORRELATION_COEFF_DEGREE = 1
CORRELATION_COEFF_NUM_POINTS = 50
GROUP_CORRELATED = True
GROUP_SIZE_THRESHOLD = 4.0/23.0
ATR_WINDOW_SIZE = 14
PREDICTION_CHANGE_THRESHOLD = 0.0001

periods = [1, 5, 15, 30, 60]
#periods = [60]

intervals = ["1min", "5min", "15min", "30min", "60min"]
#intervals = ["1min"]

#pairs = ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD"]
pairs = ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD", 
        "USDCAD", "AUDCAD", "NZDUSD", "AUDUSD", "EURJPY", "EURGBP", "CADCHF", "CHFJPY", "EURNZD", "GBPJPY"]
ALL_PAIRS = pairs
PAIRS_NUMBER = len(pairs)

CURRENCY_WEIGHTS = {
        "CHF":1, 
        "GBP":1, 
        "USD":1, 
        "EUR":1, 
        "CAD":1,
        "JPY":1, 
        "AUD":1, 
        "NZD":1 
        }


##### Interesting Combinations #####
def INDEPENDENTS():
    return ["USDCHF", "EURAUD", "NZDJPY", "CADCHF"];
def GROUP13():
    return ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD"]
def ALLINCLUDING(currencies):
    results = []
    for pair in ALL_PAIRS:
        for c in currencies:
            if c in pair:
                results.append(pair)
                break
    return results

def ALLEXCLUDING(currencies):
    results = []
    for pair in ALL_PAIRS:
        for c in currencies:
            if not c in pair:
                results.append(pair)
                break
    return results


pairCombinations = [
        INDEPENDENTS(), 
        GROUP13(), 
        ALLINCLUDING(["GBP"]), 
        ALLINCLUDING(["USD"]), 
        ALLINCLUDING(["EUR"]), 
        ALLEXCLUDING(["JPY"])]

##### Helper Methods #####
def decompose_matrix(M):
    '''
    Decompose the given matrix using eigen values and eigen vectors.
    '''
    MA = np.array(M)
    w, v = LA.eig(MA)
    W = np.diag(w)
    v_1 = LA.inv(v)
    return v, W, v_1

def calc_eigen(M, index):
    '''
    Calculates the 'index'th eigen value and eigen vector
    '''
    v, w, v_1 = decompose_matrix(M)
    vec = []
    for i in range(0,len(v)):
        vec.append(v[i][index])
    return vec, w[index][index]

def predict_next_linear(sequence):
    '''
    Predicts the next values using linear regression.
    '''
    any_complex = False
    for s in sequence:
        if isinstance(s, complex):
            any_complex = True
            break
    if any_complex:
        new_seq = []
        for s in sequence:
            if isinstance(s, complex):
                new_seq.append(s)
            else:
                new_seq.append(complex(s, 0))
        sequence = new_seq
    if any_complex:
        seq_reals = [seq.real for seq in sequence]
        seq_imags = [seq.imag for seq in sequence]
        return complex(predict_next_linear(seq_reals), predict_next_linear(seq_imags))
    else:
        x = np.array([i for i in range(0, len(sequence))]).reshape((-1, 1))
        y = np.array(sequence)
        model = LinearRegression().fit(x, y)
        return model.predict([[-1]])[0]

def find_common_pairs(charts):
    '''
    Finds the common pairs between the given charts.
    '''
    pairs = charts[0].get_pairs()
    for chart in charts:
        pairs = chart.common_pairs(pairs)
    return pairs


def pairs_other_currency(pair, c):
    '''
    Returns the other currency in the given pair
    '''
    if pair.find(c) == 0:
       return pair[3:]
    else:
       return pair[0:3]

def calculate_other_currency(pair, info, known_currency, known_value):
    '''
    Given the value of the pair and one of the currencies, returns the value of the other one.
    '''
    if not known_currency in pair:
       return None
    result = PairInfo(0,0,0)
    other_currency = pair[0:3]
    if pair.find(known_currency) == 0:
       result.value = known_value.value / info.value
       result.high = known_value.high / info.high
       result.low = known_value.low / info.low
       other_currency = pair[3:]
    else:
       result.value = known_value.value * info.value
       result.high = known_value.high * info.high
       result.low = known_value.low * info.low
    return other_currency, result

def find_currency_values(currencies, point, currency):
    '''
    Calculates the values of all currencies from pair values in the given point,
    assuming that the value of the given currency is always 1.
    '''
    currency_values = {c:None for c in currencies}
    currency_values[currency] = PairInfo(1,1,1)
    while True:
       has_none = False
       for c,v in currency_values.items():
           if v == None:
               has_none = True
               break
       if not has_none:
           break
       for pair, info in point.y.pairs.items():
           for c,v in currency_values.items():
               if v != None and c in pair:
                   c_other, v_other = calculate_other_currency(pair, info, c, v)
                   if currency_values[c_other] == None:
                       currency_values[c_other] = v_other
    return currency_values

def print_matrix(M, rows = None, cols = None):
    '''
    Prints 'rows' of rows and 'cols' number of columns of the given matrix.
    '''
    if not rows:
        rows = len(M)
    elif rows >= len(M):
        rows = len(M)
    if not cols:
        cols = len(M[0])
    elif cols >= len(M[0]):
        cols = len(M[0])
    result = ""
    for r in range(0,rows):
        for c in range(0,cols):
            result += str(M[r][c]) + "\t"
        result += "\n"
    return result


class PredictionApproach(Enum):
    ALL_EIGENS = 1
    FIRST_EIGEN_ONLY = 2

class ErrorNeutralizationApproach(Enum):
    MULTIPLICATION = 1

class PairPrediction:
    '''
    Class represents the information predicted for a specific pair
    '''
    def __init__(self, value):
        self.value = value

    def init_info(self, last_value):
        self.up = (self.value >= last_value[0])
        if abs(self.value - last_value[0]) < PREDICTION_CHANGE_THRESHOLD:
            self.up = None
        self.rate = self.value / last_value[0]
        self.change_rate = abs((self.value - last_value[0]) / last_value[0])
        self.change_range = abs(10000 * (self.value - last_value[0]))
        self.regression = predict_next_linear(last_value)
        self.regress_up = (self.regression >= last_value[0])
        if abs(self.regression - last_value[0]) < PREDICTION_CHANGE_THRESHOLD:
            self.regress_up = None
        self.regress_rate = self.regression / last_value[0]


class MatrixDecomposition:
    '''
    Class used to keep eigen values and eigen vectors of a matrix
    '''
    def __init__(self, M = None):
        if not M:
            self.M = None
            self.eigens = []
            return
        self.M = M
        self.eigens = []
        vector, w, _ = decompose_matrix(self.M)
        eig_count = len(w)
        for i in range(0, eig_count):
            eigen_value = w[i][i]
            eigen_vector = []
            for j in range(0, eig_count):
                eigen_vector.append(vector[j][i])
            eig = (eigen_value, eigen_vector)
            self.eigens.append(eig)

    def __str__(self):
        result = "Decomposing matrix:\n"
        result += print_matrix(self.M, cols = 5)
        result += "Eigen components are:\n"
        for eigen in self.eigens:
            result += str(eigen) + "\n"
        result += "\n"
        return result

    def add_component(self, value, vector):
        '''
        Adds an eigen component to the list of eigens
        '''
        self.eigens.append((value, vector))
    
    def matrix(self):
        '''
        Returns the matrix of the chart that is used to compute the eigen components in this dicomposition
        '''
        return self.M

    def recalc_matrix(self):
        '''
        Uses the eigen components to calculate the original matrix
        '''
        w = np.diag([e[0] for e in self.eigens])
        v = np.array([e[1] for e in self.eigens]).transpose()
        v_1 = LA.inv(v)
        self.M = np.matmul(np.matmul(v, w), v_1).tolist()

    def multiply_dimension_vector_by_scalar(self, dim, scalar):
        '''
        Multiplies the vector by the given scalar value
        '''
        self.eigens[dim] = (self.eigens[dim][0],[v*scalar for v in self.eigens[dim][1]])

    def get_dimension_count(self):
        '''
        Returns the number of eigen components
        '''
        return len(self.eigens)

    def get_eigen_by_dimension(self, dim):
        '''
        Returns the 'dim'th eigen value and vector
        '''
        return self.eigens[dim][0], self.eigens[dim][1]

    def apply_sign_on_dimension(self, dim, sign):
        '''
        If sign is True, makes sure the eigen value of the given dimension is greater than or equal to zero
        If false, makes sure it's less than zero.
        Note: you can multiple an eigen value by some factor and you just have to divide the corresponding
              eigen vector by that factor.
        '''
        _, vector = self.get_eigen_by_dimension(dim)
        self_sign = (vector[0] >= 0)
        if sign == self_sign:
            return
        else:
            # multiply the vector by a scalar -1
            self.multiply_dimension_vector_by_scalar(dim, -1)

class ChartDecomposition:
    '''
    Class representing the decomposition of several matrices taken from a chart.
    '''
    def __init__(self):
        self.matrix_decompositions = []
        self.chart = None

    def mds(self):
        '''
        An alias for the member matrix_decompositions
        '''
        return self.matrix_decompositions

    def init_from_chart(self, chart, from_currency=None):
        '''
        Initializes the structure from the data of the given chart
        '''
        # choose the chart to use
        if from_currency != None:
            chart = chart.materialize_by_currency(from_currency)
        self.chart = chart
        # move on windows and add matrices
        WINDOW_SIZE = len(chart.get_pairs())
        MOVE_STEP = 1
        STEPS = 20#len(chart.get_pairs())
        STEP = 0
        while STEP < STEPS:
            M = chart.matrix(WINDOW_SIZE, _offset=STEP * MOVE_STEP)
            self.add_matrix(M)
            STEP += 1
        # do required post processing tasks
        self.co_sign()

    def co_sign(self):
        '''
        Makes sure all corresponding eigen vectors have the same sign
        '''
        dimensions = self.get_dimension_count()
        for d in range(0, dimensions):
            # choose sign from the first decomposition's eigen vector's first element
            _, vector = self.mds()[0].get_eigen_by_dimension(d)
            sign = (vector[0] >= 0)
            # move on all decompositions and apply sign
            for md in self.mds():
                md.apply_sign_on_dimension(d, sign)
        #for md in self.mds():
        #    print(md)
        #print("")

    def add_matrix(self, M):
        '''
        Adds the decomposition of the given matrix
        '''
        self.matrix_decompositions.append(MatrixDecomposition(M))

    def get_matrix(self, at=0):
        '''
        Returns the matrix of the MatrixDecomposition at location 'at'. If no 'at' is given, returns the first inserted matrix
        or in other words the newest sampling matrix.
        '''
        return self.mds()[at].matrix()

    def get_dimension_count(self):
        '''
        Returns the number of eigen components that are computed from given matrices
        '''
        if len(self.matrix_decompositions) == 0:
            return None
        return self.matrix_decompositions[0].get_dimension_count()

    def get_eigen_values_of_dimension(self, dim):
        '''
        Returns the list of the 'dim'th eigen values from all given matrices
        '''
        values = []
        for md in self.matrix_decompositions:
            v, _ = md.get_eigen_by_dimension(dim)
            values.append(v)
        return values

    def get_kth_element_of_eigen_vectors_of_dimension(self, dim, k):
        '''
        Returns the list of the 'dim'th eigen values from all given matrices
        '''
        values = []
        for md in self.matrix_decompositions:
            _, vec = md.get_eigen_by_dimension(dim)
            values.append(vec[k])
        return values

    def predict_eigen_value_of_dimension(self, dim):
        '''
        Returns a linear prediction on the 'dim'th eigen value of the expected next matrix
        '''
        eigen_values = self.get_eigen_values_of_dimension(dim)
        return predict_next_linear(eigen_values)

    def predict_kth_element_of_eigen_vector_of_dimension(self, dim, k):
        '''
        Returns a linear prediction on the 'dim'th eigen value of the expected next matrix
        '''
        values = self.get_kth_element_of_eigen_vectors_of_dimension(dim, k)
        return predict_next_linear(values)

    def predict_eigen_vector_of_dimension(self, dim):
        '''
        Returns the 'dim'th eigen vector of the expected next matrix
        '''
        return [self.predict_kth_element_of_eigen_vector_of_dimension(dim, k) for k in range(0, self.get_dimension_count())]

    def predict_eigen(self, dimension=0):
        '''
        Returns the predicted eigen value and vector of the given dimension
        '''
        return (self.predict_eigen_value_of_dimension(dimension), 
                self.predict_eigen_vector_of_dimension(dimension))

    def predict_eigens(self, dimensions = None):
        '''
        Returns the predicted eigen values and vectors of all dimensions
        '''
        result = {}
        for dim in range(0, self.get_dimension_count()):
            if dimensions != None and dim not in dimensions:
                continue
            result[dim] = (predict_eigen_value_of_dimension(dimension), predict_eigen_vector_of_dimension(dimension))
        return result
    def predict_expected_matrix_decomposition(self):
        '''
        Returns the MatrixDecomposition object of the next expected matrix
        '''
        md = MatrixDecomposition()
        # fill eigen values
        predicted_eigens = self.predict_eigens()
        for dim in range(0, self.get_dimension_count()):
            (value, vector) = predicted_eigens[dim]
            md.add_component(value, vector)
        # calculate matrix
        md.recalc_matrix()
        return md

    def predict_next_matrix(self, approach = PredictionApproach.FIRST_EIGEN_ONLY):
        '''
        Predicts/Returns the next expected matrix
        '''
        if approach == PredictionApproach.ALL_EIGENS:
            return self.predict_next_matrix_decomposition().M
        elif approach == PredictionApproach.FIRST_EIGEN_ONLY:
            return self.build_next_matrix()

    def build_next_matrix(self):
        '''
        Places the last 12 columns of the last matrix (**which is added first**) next to a predicted column 
        which is predicted using linear regression on the first eigen value and vector.
        '''
        # get the last len(dimensions)-1 columns of sampling (for example last 12 columns for a set of 13 pairs)
        M = self.get_matrix()
        ## remove last column of M
        M = [r[:len(r)-1] for r in M]
        # the formula is basically this: ([x] + M[i]) . (vector)       =   value * vector[i]
        #                            ->  x*vector[0] + M[i].vector[1:] =   value * vector[i]
        #                            ->                             x  =   (value * vector[i] - M[i].vector[1:]) / vector[0]
        predictions = []
        (eig_value, eig_vector) = self.predict_eigen()
        for i, row in enumerate(M):
            predictions.append( (eig_value * eig_vector[i] - np.dot(np.array(row), np.array(eig_vector[1:]))) / eig_vector[0] )
        # append prediction to M
        new_M = [[predictions[i]]+row for i,row in enumerate(M)]
        # neutralize predictions's error by multiplication
        new_M = self.neutralize_base_error(new_M)
        return new_M

    def neutralize_base_error(self, M, approach=ErrorNeutralizationApproach.MULTIPLICATION):
        '''
        if all values in any row of M are X, scale the predicted column so the value in that row stays X
        '''
        if approach == ErrorNeutralizationApproach.MULTIPLICATION:
            # find which row has all same values (excluding the first element)
            all_one_row_index = None
            all_one_row_value = None
            for i,row in enumerate(M):
                all_one = True
                check_value = row[0]
                for v in row[1:]:
                    if v != check_value:
                        all_one = False
                        break
                if all_one:
                    all_one_row_index = i
                    all_one_row_value = check_value
                    break
            if all_one_row_index == None:# there is no constant row so no need for scaling
                return M
            else:
                # calculate the factor
                scale_factor = all_one_row_value / M[all_one_row_index][0]
                # apply the factor
                for i in range(0, len(M)):
                    M[i][0] *= scale_factor
                return M
        else:
            return M # for now

    def predict_next(self, approach=PredictionApproach.FIRST_EIGEN_ONLY):
        '''
        Predicts the next value of each pair using our approach.
        '''
        next_M = self.predict_next_matrix(approach)
        predictions = []
        for row in next_M:
            p = PairPrediction(row[0])
            p.init_info(row[1:])
            predictions.append(p)
        return predictions

##########################


class PairInfo:
    '''
    Class representing information of a point for a specific pair.
    '''
    def __init__(self, value, high, low):
        self.value = value
        self.high = high
        self.low = low

    def normalize(self, factor):
        '''
        Normalize the pair info by multiplying it into the given factor.
        '''
        self.value = self.value * factor
        self.high = self.high * factor
        self.low = self.low * factor

    def denormalize(self, factor):
        '''
        Denormalize the pair info by diving it by the given factor. This method is intended to undo `normalize`
        '''
        self.value = self.value / factor
        self.high = self.high / factor
        self.low = self.low / factor


    def logarithm(self):
        '''
        Take logarithm of all values in the pair.
        '''
        self.value = np.log(self.value)
        self.high = np.log(self.high)
        self.low = np.log(self.low)

    def range(self, prevPairInfo):
        '''
        Returns the change range of this pair info.
        '''
        v1 = int(self.high * 10000 - self.low * 10000)
        v2 = abs(int(self.high * 10000 - prevPairInfo.value * 10000))
        v3 = abs(int(self.low * 10000 - prevPairInfo.value * 10000))
        return max(v1, v2, v3)

    def write_to_file(self, f):
        '''
        Write pair info to the given file
        '''
        f.write(str(self.value) + '\n')
        f.write(str(self.high) + '\n')
        f.write(str(self.low) + '\n')

    def read_from_file(self, f):
        '''
        Read pair info from the given file.
        '''
        self.value = float(f.readline())
        self.high = float(f.readline())
        self.low = float(f.readline())


class PointInfo:
    '''
    Class representing the information in a point, which consists of the information of several pairs.
    '''
    def __init__(self):
        self.pairs = {}
        self.timestamp = None

    def add_pair(self, pair, pairInfo):
        '''
        Add a new pair info to the point.
        '''
        self.pairs[pair] = pairInfo

    def get_pair_at(self, pairIndex):
        '''
        Returns the information of the 'pairIndex'th pair in the point.
        '''
        for i,p in enumerate(self.pairs.keys()):
            if i == pairIndex:
                return p
        return None

    def common_pairs(self, pairs):
        '''
        Returns the intersection of the given set of pairs and the pairs kept in this point.
        '''
        result = []
        for p in pairs:
            if p in self.pairs:
                result.append(p)
        return result

    def write_to_file(self, f):
        '''
        Writes point data to the given file.
        '''
        f.write(str(self.timestamp) + '\n')
        f.write(str(len(self.pairs)) + '\n')
        for pair, pairInfo in self.pairs.items():
            f.write(str(pair) + '\n')
            pairInfo.write_to_file(f)

    def read_from_file(self, f):
        '''
        Reads the data of this point from the given file.
        '''
        timestamp_str = f.readline().strip()
        self.timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        pairs_count = int(f.readline())
        for i in range(0, pairs_count):
            pair = f.readline().strip()
            pairInfo = PairInfo(0,0,0)
            pairInfo.read_from_file(f)
            self.pairs[pair] = pairInfo

class Point:
    '''
    A chart point which, consists of x which is a number and y which is points information.
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(x) + ", " + str(y) + ")"

    def normalize(self, factors):
        '''
        Normalize all pair values of this point using the given pair to factor map.
        '''
        for pair,info in self.y.pairs.items():
            info.normalize(factors[pair])

    def denormalize(self, factors):
        '''
        Denormalize all pair values of this point using the given pair to factor map.
        '''
        for pair,info in self.y.pairs.items():
            info.denormalize(factors[pair])

    def logarithm(self):
        '''
        Take logarithm of the information of all pairs in the point
        '''
        for _,info in self.y.pairs.items():
            info.logarithm()

    def write_to_file(self, f):
        '''
        Write the data of this point to the given file.
        '''
        f.write(str(self.x) + '\n')
        self.y.write_to_file(f)

    def read_from_file(self, f):
        '''
        Read the data of this point from the given file.
        '''
        self.x = int(f.readline())
        new_point_info = PointInfo()
        new_point_info.read_from_file(f)
        self.y = new_point_info

class PairCorrelation:
    '''
    Class representing the correlation between a number of pairs (usually two)
    '''
    def __init__(self, pairs, coeff):
        self.pairs = pairs
        self.coeff = coeff

    def __str__(self):
        return str(self.pairs) + str("/") + str(self.coeff)

class Chart:
    '''
    Class representing a series of points which are samples taken for some pairs with a certain period of time.
    '''
    def __init__(self, sampling_period):
        self.period = sampling_period
        self.points = []
        self.norm_factors = None
        self.offset = 0

    def __str__(self):
        return str(self.period) + "\t\t" + str(len(self.points))

    def set_offset(self, offset):
        self.offset = offset

    def add_point(self, seq_number, point_value, SCALE_X=True):
        '''
        Adds a point whose sequence number in the chart is given.
        '''
        x_value = 0
        if SCALE_X:
            x_value = self.seq_number_to_x_value(seq_number)
        else:
            x_value = seq_number
        point = Point(x_value, point_value)
        self.points.append(point)

    def seq_number_to_x_value(self, seq_number):
        '''
        Calculates the x value of a point from the sequence number
        '''
        return -1 * seq_number * self.period

    def x_value_to_seq_number(self, x):
        '''
        Calculates the sequence number of a point from its x value
        '''
        return x / (-1 * self.period)

    def clean(self):
        '''
        Makes sure all points in the chart have the same set of pairs
        '''
        pairs = self.get_pairs()
        points = []
        # Note: even if offset is set, we clean all the chart here.
        for point in self.points:
            if len(point.y.pairs) == len(pairs):
                points.append(point)
            else:
                break
        self.points = points

    def is_usable(self):
        '''
        Returns true if chart has at least two points, and false otherwise
        '''
        return (len(self.points) - self.offset) >= 2

    def size(self):
        '''
        Returns the number of points in the chart.
        '''
        return len(self.points) - self.offset

    def get_last_point(self):
        '''
        Returns the last point in the chart (the newest point, the point with seq_number zero).
        '''
        return self.points[0 + self.offset]

    def get_pairs(self):
        '''
        Returns a list of all pairs in the chart.
        '''
        pairs = []
        for p,_ in self.get_last_point().y.pairs.items():
            pairs.append(p)
        return pairs

    def decompose(self):
        M = self.matrix(len(self.get_pairs()))
        v, w, _ = decompose_matrix(M)
        #print(v)
        #print(w)
        dec = MatrixDecomposition(M)
        #print(str(dec))
    def normalize(self):
        '''
        Normalize the chart data.
        '''
        # 1. Calc norm factor for each pair
        normFactors = {}
        for pair in self.get_pairs():
            var = self.pair_data_variance(pair)
            avg = self.pair_data_average(pair)
            factor = var / avg
            normFactors[pair] = factor
        # 2. normalize each point with the calculated factors
        for point in self.points[self.offset:]:
            point.normalize(normFactors)
        # 3. Update the normalization factors kept in the chart
        if self.norm_factors == None:
            self.norm_factors = normFactors
        else:
            newNormFactors = []
            for p, nf in normFactors.items():
                newNormFactors.append(self.norm_factors[p] * nf)
            self.norm_factors = newNormFactors

    def denormalize(self):
        '''
        Denormalize the chart data.
        '''
        if self.norm_factors == None:
            return
        for point in self.points[self.offset:]:
            point.denormalize(self.norm_factors)
        self.norm_factors = None

    def logarithm(self):
        '''
        Take logarithm of all the Y values of the chart points.
        '''
        for point in self.points[self.offset:]:
            point.logarithm()

    def sub_chart(self, pairs):
        '''
        Return a chart instance with only pairs that are given.
        '''
        result = Chart(self.period)
        # Iterate on all points and only keep the intended pairs
        for point in self.points[self.offset:]:
            seq_num = point.x / (-1 * self.period)
            pointValue = PointInfo()
            for pair in pairs:
                if pair in point.y.pairs:
                    pairInfo = point.y.pairs[pair]
                    pointValue.add_pair(pair, pairInfo)
            result.add_point(seq_num, pointValue)
        return result

    def common_pairs(self, pairs):
        '''
        Returns the intersection of chart's pairs with the given set of pairs
        '''
        for point in self.points[self.offset:]:
            pairs = point.y.common_pairs(pairs)
        return pairs

    def get_correlation_coefficient(self, degree=1):
        '''
        Returns the correlation matrix of the chart's pair values. It takes correlation 'degree' number of time.
        '''
        values = np.array(self.matrix(CORRELATION_COEFF_NUM_POINTS))
        for i in range(0,degree):
            values = np.cov(values)
        return values

    def modularize(self):
        '''
        Finds the groups of pairs that have the most correlation among the members of the group. Finds the components of the pair set of the chart.
        '''
        # 1. Find correlation coefficients
        coeffs = self.get_correlation_coefficient(CORRELATION_COEFF_DEGREE).tolist()
        pairs = self.get_pairs()
        # 2. Find the correlation pairs that are going to be kept
        relations = []
        for i, row in enumerate(coeffs):
            for j, coeff in enumerate(row):
                if j <= i:
                    continue
                relations.append(PairCorrelation([pairs[i],pairs[j]], coeff))
        CUT_THRESHOLD = CORRELATION_PRUNE_FACTOR
        while True:
            groups = self.modularize_based_on_cut_threshold(relations, CUT_THRESHOLD, GROUP_CORRELATED)
            if groups == None:
                return [self.get_pairs()]
            if len(groups) > 1:
                return groups
            else:
                CUT_THRESHOLD **= 2

    def modularize_based_on_cut_threshold(self, relations, prune_factor, group_correlated):
        '''
        Finds connected components based on the given pair-wise relations and the given cut threshold
        '''
        # 1. Sort the pair-wise correlations
        relations.sort(key=lambda x: x.coeff, reverse=group_correlated)
        # 2. Prune the relations
        relations = relations[0:int(prune_factor * len(relations))]
        if len(relations) == 0:
            return None
        # 3. Find separate components (called groups here) based on the 2-pair relations
        pairs = self.get_pairs()
        pairGroups = {p:None for p in pairs}
        def find_root(pair,graph):
            while True:
                if graph[pair] == None:
                    return pair
                else:
                    pair = graph[pair]
        for r in relations:
            root1 = find_root(r.pairs[0], pairGroups)
            root2 = find_root(r.pairs[1], pairGroups)
            if root1 != root2:
                pairGroups[root1] = root2
        roots = {}
        for pair in pairGroups:
            root = find_root(pair, pairGroups)
            if root in roots:
                roots[root].append(pair)
            else:
                roots[root] = [pair]
        # 4. Return only those groups that are large enough
        groups = []
        for root, group in roots.items():
            if len(group) >= GROUP_SIZE_THRESHOLD * len(pairs):
                groups.append(group)
        return groups

    def predict_first_eigen(self):
        '''
        Predicts the next value of the first eigen value and vector
        '''
        N = len(self.get_pairs())
        # calculate the last N eigen values and vectors
        eigenValues = []
        eigenVectors = []
        for pairIndex in range(0, N):
            eigenVectors.append([])
        for offset in range(0, N):
            M = self.matrix(N, offset)
            vec, w = calc_eigen(M, 0)
            # add w to the list to be used for regression
            eigenValues.append(w.real)
            # add vec elements to the list to be used for regression
            for i, vec_element in enumerate(vec):
                eigenVectors[i].append(vec_element.real)
        # predict w
        w = predict_next_linear(eigenValues)
        # predict vector
        vec = []
        for v in eigenVectors:
            v_pred = predict_next_linear(v)
            vec.append(v_pred)
        return vec, w

    def predict_first_eigen_value(self):
        '''
        Returns the prediction for the first eigen value
        '''
        _, w = self.predict_first_eigen()
        return w

    def pair_values(self, pair):
        '''
        Returns an array of the point values of the given pair
        '''
        pairResult = []
        for i in range(self.offset,len(self.points)):
            pairResult.append(self.points[i].y.pairs[pair].value)
        return pairResult

    def pair_data_variance(self, pair):
        '''
        Returns the data variance of the point values of the given pair.
        '''
        values_ = self.pair_values(pair)
        values = np.array(values_)
        return np.var(values, dtype=np.float64)

    def pair_data_average(self, pair):
        '''
        Returns the data average of the point values of the given pair.
        '''
        values_ = self.pair_values(pair)
        values = np.array(values_)
        return np.mean(values, dtype=np.float64)

    def matrix(self, N, _offset=0):
        '''
        Returns a matrix with the N values of all pairs starting from data point at the offset. 
        Each row is the last N values of a pair. So the returning matrix is P x N where P is the number of pairs and N is given.
        '''
        result = []
        for pair in self.get_pairs():
            pairResult = []
            for i in range(0,N):
                pairResult.append(self.points[self.offset + i + _offset].y.pairs[pair].value)
            result.append(pairResult)
        return result

    def range_matrix(self, N, _offset=0):
        '''
        Returns the matrix of changes in the N points with the given offset from the last.
        '''
        result = []
        for pair in self.get_pairs():
            pairResult = []
            for i in range(0,N):
                pairResult.append(self.points[self.offset + i + _offset].y.pairs[pair].range(self.points[ self.offset + i + _offset+1].y.pairs[pair]))
            result.append(pairResult)
        return result

    def average_recent_range(self):
        '''
        Returns the value of ATR indicator for all pairs.
        '''
        result = {}
        pairs = self.get_pairs()
        M = self.range_matrix(ATR_WINDOW_SIZE)
        for i, p in enumerate(pairs):
            result[p] = int(sum(M[i]) / len(M[i]))
        return result

    def predict_next_change_range(self):
        '''
        Returns the predicted next change for each pair
        '''
        predictions = self.predict_next()
        C = self.matrix(1)
        results = {}
        i = 0
        for pair, pred in predictions.items():
            results[pair] = abs(int(10000 * pred - 10000 * C[i][0]))
            i += 1
        return results

    def predict_next_regression(self, window_size=None):
        '''
        Predicts the next value of each pair using regression.
        '''
        pairs = self.get_pairs()
        if window_size == None:
            window_size = len(pairs)
        M = self.matrix(window_size)
        result = {}
        for i, pair in enumerate(pairs):
            result[pair] = predict_next_linear(M[i])
        return result

    def predict_next_binary_regression(self):
        '''
        For each predicts if the next value is higher or lower than the last value, using regression for prediction.
        '''
        return self.convert_prediction_to_binary(self.predict_next_regression())

    def predict_next(self):
        '''
        Predicts the next value of each pair using our approach.
        '''
        pairs = self.get_pairs()
        vec, w = self.predict_first_eigen()
        M = self.matrix(len(pairs)-1)
        predictions = []
        for i, seq in enumerate(M):
            dotSum = 0
            for seq_i in range(0, len(seq)):
                dotSum += seq[seq_i] * vec[seq_i+1]
            prediction = ((w * vec[i]) - dotSum) / vec[0]
            predictions.append(prediction)
        results = {}
        for i, pair in enumerate(pairs):
            results[pair] = predictions[i]
        return results

    def predict_next_binary(self):
        '''
        For each pair predicts whether the next value is higher than the last value or lower. True if higher and false otherwise
        '''
        return self.convert_prediction_to_binary(self.predict_next())

    def convert_prediction_to_binary(self, predictions):
        '''
        Converts a prediction to binary results: True if the value is higher than the last value and False otherwise
        '''
        C = self.matrix(1)
        results = {}
        i = 0
        for pair, pred in predictions.items():
            if pred > C[i][0]:
                results[pair] = True
            else:
                results[pair] = False
            i += 1
        return results

    def get_currencies(self):
        '''
        Returns the set of all available currencies in the list of pairs in this chart.
        '''
        currencies = set()
        for pair in self.get_pairs():
            currencies.add(pair[0:3])
            currencies.add(pair[3:])
        return currencies

    def materialize_by_currency(self, currency):
        '''
        Returns a Chart object which offers currencies instead of pairs. In these objects, pair API return and use currency values.
        Currency values are assumed with the assumption that 'currencies''s value is always 1. currency is a three character currency name string.
        '''
        # find out what currencies do we have
        currencies = self.get_currencies()
        # for each point, calculate currency values and create a new point
        result = Chart(self.period)
        for point in self.points[self.offset:]:
            # find currency values from point pair values
            currency_values = find_currency_values(currencies, point, currency)
            # create corresponding currency point
            pointInfo = PointInfo()
            for c,v in currency_values.items():
                pointInfo.add_pair(c, v)
            result.add_point(point.x, pointInfo, SCALE_X=False)
        return result

    def process_chart_currency_decompositions(self):
        #print("Processing chart(" + str(self.period) + "):")
        currencies = self.get_currencies()
        # for each currency make it base
        chart_decompositions = {}
        for currency in currencies:
            #print("-- using currency " + currency + " as the base:")
            cd = ChartDecomposition()
            cd.init_from_chart(self, from_currency=currency)
            chart_decompositions[currency] = cd
        return chart_decompositions

    def write_to_file(self, f):
        '''
        Writes the chart data into the given file.
        '''
        f.write(str(self.period) + '\n')
        f.write(str(len(self.points)) + '\n')
        for point in self.points:
            point.write_to_file(f)

    def read_from_file(self, f):
        '''
        Reads the chart data from the given file.
        '''
        self.period = int(f.readline())
        points_count = int(f.readline())
        for i in range(0, points_count):
            new_point = Point(0, 0)
            new_point.read_from_file(f)
            self.points.append(new_point)

def calc_pair_from_currencies(pair, currencies_predictions):
    c1 = pair[0:3]
    c2 = pair[3:]
    return currencies_predictions[c1] / currencies_predictions[c2]

def prepare_charts_to_use(file_path, test=False):
    '''
    Returns a list of Chart objects that are read from the given file and prepared for use.
    '''
    global periods, intervals, pairs
    charts = []
   
    # read file
    charts_file = open(file_path, "r")
    charts2 = []
    charts_count = int(charts_file.readline())
    
    for i in range(0, charts_count):
        chart = Chart(0)
        chart.read_from_file(charts_file)
        charts2.append(chart)
    
    charts_file.close()
    
    
    # clean the charts (remove all points with info for only some of the pairs)
    for chart in charts2:
        chart.clean()
    
    # make sure all charts have the same set of pairs
    charts_tmp = []
    common_pairs = find_common_pairs(charts2)
    for chart in charts2:
        charts_tmp.append(chart.sub_chart(common_pairs))
    charts2 = charts_tmp
    pairs = common_pairs
    
    # final checks and post processings on the charts
    new_periods = []
    new_intervals = []
    for i, chart in enumerate(charts2):
        if chart.is_usable():
            #chart.logarithm()
            charts.append(chart)
            new_periods.append(periods[i])
            new_intervals.append(intervals[i])
    
    periods = new_periods
    intervals = new_intervals
    
    print("Period(min)\t# of points")
    for chart in charts:
        print(chart)
    print("-----------------------------------")
   
    print("\033[1;33;40m\n")
    print("Period\t",end='')
    for p in charts[0].get_pairs():
        print(str(p) + "\t", end='')
    print("")
    print("\033[0;37;40m")
    for chart in charts:
        decompositions = chart.process_chart_currency_decompositions()
        # calculate aggregated predictions for currencies
        currencies = None
        currencies_sums = None
        for currency, decomp in decompositions.items():
            if currencies == None:
                currencies = decomp.chart.get_pairs()
                currencies_sums = {p:0 for p in decomp.chart.get_pairs()}
            predictions = decomp.predict_next()
            for i,p in enumerate(predictions):
                predicted_currency = currencies[i]
                currencies_sums[predicted_currency] += CURRENCY_WEIGHTS[predicted_currency] * p.value
        # calculate predicted value for all pairs
        pair_predictions = {}
        for pair in chart.get_pairs():
            pair_predictions[pair] = PairPrediction(calc_pair_from_currencies(pair, currencies_sums))
        last_few_columns = chart.matrix(len(chart.get_pairs()))
        for i, p in enumerate(pair_predictions):
            pair_predictions[p].init_info(last_few_columns[i])
        print("\033[1;37;40m" + str(chart.period) + "min\033[0;37;40m\t",end='')
        for p, prediction in pair_predictions.items():
            print(("\033[1;34;40m B" if prediction.up == True else ( "\033[1;31;40m S" if prediction.up == False else "\033[1;31;40m -") ) + "\033[0;33;40m / \033[0;37;40m" + ("B" if prediction.regress_up else "S") + "\t", end='')
        print("")
    if not test:
        return charts
    # tests and experiments


#charts = prepare_charts_to_use("charts.index", test=True)
