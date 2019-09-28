#!/bin/python

import requests
import json
import time
import chartslib

SLEEP_TIME=13

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



charts_file = open("charts.index","w+")

period_index = 0
for period in chartslib.periods:
    print("Fetching for period " + str(period))
    interval = chartslib.intervals[period_index]
    chart = chartslib.Chart(period)
    pairValues = {}
    for pair in chartslib.pairs:
        print("\t Fetching pair " + str(pair))
        pairValues[pair] = []
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
            pairValues[pair].append(pairInfo)
        time.sleep(SLEEP_TIME)

    sequence = {}
    # initialization
    for pair, pairSequence in pairValues.items():
        point_seq = 0
        for point in pairSequence:
            sequence[point_seq] = chartslib.PointInfo()
            point_seq += 1
    # filling it up
    for pair, pairSequence in pairValues.items():
        point_seq = 0
        for pairInfo in pairSequence:
            sequence[point_seq].addPair(pair, pairInfo)
            point_seq += 1
    # initializing the chart
    for seq_num, seq_point in sequence.items():
        chart.addPoint(seq_num, seq_point)
    charts.append(chart)
    period_index += 1

charts_file.write(str(len(charts)) + '\n')
for chart in charts:
    chart.writeToFile(charts_file)
charts_file.close()
