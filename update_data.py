#!/bin/python

import requests
import json
import time
import chartslib
import shutil
from datetime import datetime

SLEEP_TIME=13

def file_name(prefix):
    return prefix + datetime.now().strftime("%b_%d_%Y_%H_%M")

def parse_timestamp(time_str):
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

charts = []


#charts_file = open("charts.index", "r")
#charts2 = []
#charts_count = int(charts_file.readline())
#
#for i in range(0, charts_count):
#    chart = Chart(0)
#    chart.readFromFile(charts_file)
#    charts2.append(chart)
#
#charts_file.close()

try:
    shutil.move('charts.index', file_name('charts.index.'))
except FileNotFoundError:
    pass


charts_file = open("charts.index","w+")

period_index = 0
for period in chartslib.periods:
    print("Fetching for period " + str(period))
    interval = chartslib.intervals[period_index]
    chart = chartslib.Chart(period)
    pairValues = {}
    for pair in chartslib.pairs:
        print("\t Fetching pair " + str(pair))
        pairValues[pair] = {}
        request = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol='+pair[0:3]+'&to_symbol='+pair[3:6]+'&outputsize=full&interval='+interval+'&apikey=PEHRYBRLFJNYP2IK&datatype=json'
        print('\t' + request)
        rates_response = requests.get(request)
        if rates_response.status_code != 200:
            continue
        try:
            candle_array = json.loads(str(rates_response.content)[2:-1].replace("\\n", ""))['Time Series FX ('+interval+')']
        except KeyError:
            print(rates_response.content)
            time.sleep(SLEEP_TIME)
            continue
        for timestamp, info in candle_array.items():
            pairInfo = chartslib.PairInfo(info["1. open"], info["2. high"], info["3. low"])
            pairValues[pair][parse_timestamp(timestamp)] = pairInfo
        time.sleep(SLEEP_TIME)

    sequence = {}
    # initialization
    for pair, pairSequence in pairValues.items():
        for timestamp, _ in pairSequence.items():
            sequence[timestamp] = chartslib.PointInfo()
    # filling it up
    for pair, pairSequence in pairValues.items():
        for timestamp, pairInfo in pairSequence.items():
            sequence[timestamp].add_pair(pair, pairInfo)
    # clean it: remove those timestamps that don't have all pairs
    new_sequence = {}
    ## find the largest number of pairs per point
    pairs_count = 0
    for _,point in sequence.items():
        pairs_count = max(pairs_count, len(point.pairs))
    ## prune the points
    for timestamp, point in sequence.items():
        if pairs_count == len(point.pairs):
            new_sequence[timestamp] = point
    sequence = new_sequence
    # initializing the chart
    for seq_num, timestamp in enumerate(sequence):
        seq_point = sequence[timestamp]
        seq_point.timestamp = timestamp
        chart.add_point(seq_num, seq_point)
    charts.append(chart)
    period_index += 1

charts_file.write(str(len(charts)) + '\n')
for chart in charts:
    chart.write_to_file(charts_file)
charts_file.close()

