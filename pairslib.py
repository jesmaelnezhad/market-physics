#!/bin/python

pairs = ["USDCHF","EURUSD", "AUDCHF", "USDJPY", "GBPUSD", "AUDJPY", "EURAUD", "EURCHF", "CADJPY", "NZDJPY", "GBPCHF", "EURCAD", "AUDNZD",
        "USDCAD", "AUDCAD", "NZDUSD", "AUDUSD", "EURJPY", "EURGBP", "CADCHF", "CHFJPY", "EURNZD", "GBPJPY"]

def find_currencies(pairs):
    currencies = set()
    for p in pairs:
        c1, c2 = break_pair(p)
        currencies.add(c1)
        currencies.add(c2)
    return currencies

def break_pair(pair):
    return pair[0:3], pair[3:]

def add_to_count_map(element, count_map):
    if element in count_map:
        count_map[element] += 1
    else:
        count_map[element] = 1


currencies = find_currencies(pairs)

def has_linear_dependency(pair, pairs):
    pass

def division_equals(numinators, denominators, pair):
    # currency to its count
    numinator_currencies = {}
    denominator_currencies = {}
    for n in numinators:
        c1, c2 = break_pair(n)
        add_to_count_map(c1, numinator_currencies)
        add_to_count_map(c2, denominator_currencies)
    for d in denominators:
        c1, c2 = break_pair(d)
        add_to_count_map(c2, numinator_currencies)
        add_to_count_map(c1, denominator_currencies)
    c1, c2 = break_pair(pair)
    add_to_count_map(c2, numinator_currencies)
    add_to_count_map(c1, denominator_currencies)

    return numinator_currencies == denominator_currencies

print(division_equals(['EURGBP'], ['USDCHF', 'CHFGBP'], 'EURUSD'))
