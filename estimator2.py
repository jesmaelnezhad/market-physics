#!/bin/python
import sys
import copy

NUM_ITERATIONS=5000

class Exp:
    def __init__(self, *args):
        self.op = args[0]
        self.operands = list()
        for operand in args[1:]:
            self.operands.append(operand)

    def __str__(self):
        if len(self.operands) == 1:
            return self.op + str(self.operands[0]) 
        else:
            result = "(" + str(self.operands[0])
        for i in range(1,len(self.operands)):
            result = result + " " + self.op + " " + str(self.operands[i])
        result = result + ")"
        return result
    def eval(self, values, prevValues):
        if self.op == 'd':
            return values[self.operands[0]] - prevValues[self.operands[0]]
        elif self.op == 'v':
            return self.operands[0]
        elif self.op == 'p':
            return values[self.operands[0]]
        
        left = self.operands[0].eval(values, prevValues)
        right = self.operands[1].eval(values, prevValues)

        if self.op == '*':
            return left * right
        elif self.op == '/':
            return left / right
        elif self.op == '**':
            return left ** right
        elif self.op == '+':
            return left + right
        elif self.op == '-':
            return left - right
    def calcDerivation(self):
        if self.op == 'p':
            dSelf = copy.deepcopy(self)
            dSelf.op = 'd'
            return dSelf
        elif self.op == 'd':
            dSelf = copy.deepcopy(self)
            dSelf.op = 'dd'
            return dSelf
        elif self.op == 'v':
            return Exp('v', 0)
        elif self.op == '*':
            return Exp('+', Exp('*', self.operands[0].calcDerivation(), self.operands[1]), Exp('*', self.operands[0], self.operands[1].calcDerivation()))
        elif self.op == '+' or self.op == '-':
            return Exp(self.op, self.operands[0].calcDerivation(), self.operands[1].calcDerivation())
        elif self.op == '/':
            return Exp('/', 
                    Exp('-', 
                        Exp('*', 
                            self.operands[0].calcDerivation() , 
                            self.operands[1] ) , 
                        Exp('*', 
                            self.operands[0], 
                            self.operands[1].calcDerivation() 
                            ) 
                        ), 
                    Exp('**', self.operands[1], Exp('v', 2))
                    )
        elif self.op == '**':
            return Exp('*', Exp('v', self.operands[1]), Exp('**', self.operands[0], Exp('v', self.operands[1]-1)))




previousValues = {}
currentValues = {}
estimatedNextValues = {}

def V_t(pair):
    return currentValues[pair]

def dV_t(pair):
    return currentValues[pair] - previousValues[pair]

def V_tplus1(pair):
    return estimatedNextValues[pair]

def SetEstimated_V_tplus1(pair, value):
    estimatedNextValues[pair] = value

def dV_tplus1(pair):
    return estimatedNextValues[pair] - currentValues[pair]

def SetEstimated_dV_tplus1(pair, dvalue):
    estimatedNextValues[pair] = dvalue + currentValues[pair]

def initValues(eq):
    for pair in eq[0]:
        print(pair[0] + "/" + pair[1])
    for pair in eq[1]:
        print(pair[0] + "/" + pair[1])
    openCloseValueStrs = input("Enter open and close values in the printed order:").split(" ")
    for i in range(0,len(eq[0])):
        pair = eq[0][i]
        previousValues[pair] = float(openCloseValueStrs[2*i])
        currentValues[pair] = float(openCloseValueStrs[2*i+1])
    for i in range(0,len(eq[1])):
        pair = eq[1][i]
        previousValues[pair] = float(openCloseValueStrs[2*i])
        currentValues[pair] = float(openCloseValueStrs[2*i+1])


def setInitialEstimates(eq):
    for pair in currentValues:
        estimatedNextValues[pair] = 2 * currentValues[pair] - previousValues[pair]

def refineEstimates(eq):
    estimatedNextValuesTmp = {}
    for pair in currentValues:
        f = formula(pair, eq)
        df = f.calcDerivation()
        df_tplus1 = evalFormulaWithEstimatedValues(df)
        estimatedNextValuesTmp[pair] = currentValues[pair] + df_tplus1
    estimatedNextValues = estimatedNextValuesTmp

def parseEquation(left, right):
    leftPairStrs = left.split('*')
    rightPairStrs = right.split('*')
    resultEquation = ([],[])
    for p in leftPairStrs:
        pt, pb = p.split("/")
        resultEquation[0].append((pt,pb))
    for p in rightPairStrs:
        pt, pb = p.split("/")
        resultEquation[1].append((pt,pb))
    return resultEquation

def formula(estmtPair, eq):
   # check if (top/buttom) is on top of eq
   if estmtPair in eq[0]:
       newEq = ([],[])
       for pair in eq[1]:
           newEq[0].append(pair)
       for pair in eq[0]:
           if pair != estmtPair:
               newEq[1].append(pair)
       return eq2Exp(newEq)
   # check buttom
   if estmtPair in eq[1]:
       newEq = ([],[])
       for pair in eq[0]:
           newEq[0].append(pair)
       for pair in eq[1]:
           if pair != estmtPair:
               newEq[1].append(pair)
       return eq2Exp(newEq)
   return None

def printAll():
    print("Name\tPrevious\tCurrent\tEstimated\tPercentage of change")
    for k,v in previousValues.items():
        print(str(k[0] + "/" + k[1]) + "\t" + str(v) + "\t" \
                + str(currentValues[k]) + "\t" \
                + str(estimatedNextValues[k]) + "\t" \
                + str((estimatedNextValues[k] - currentValues[k])/currentValues[k]*100.0) + "\t" \
                + ("yes" if ((estimatedNextValues[k] - currentValues[k])*(currentValues[k]-previousValues[k])) > 0 else "no"))


def eq2Exp(eq):
    if len(eq[0]) == 0:
        return Exp('/', Exp('v', 1), listMultiplication2Exp(eq[1]))
    elif len(eq[1]) == 0:
        return listMultiplication2Exp(eq[0])
    return Exp('/', listMultiplication2Exp(eq[0]), listMultiplication2Exp(eq[1]))

def listMultiplication2Exp(pairList):
    if len(pairList) == 1:
        return Exp('p', pairList[0])
    return Exp('*', Exp('p', pairList[0]), listMultiplication2Exp(pairList[1:]))

def evalFormulaWithEstimatedValues(formula):
    return formula.eval(estimatedNextValues, currentValues)

if __name__=="__main__":
    #left = input("Enter the left side of the equation: ")
    left = "USD/CHF*EUR/USD*NZD/CHF*USD/JPY*GBP/USD*AUD/JPY*EUR/AUD"
    #left = "A/B*C/D"
    #left = "A/B*C/D*G/R"
    #right = input("Enter the right side of the equation: ")
    right = "CAD/CHF*CAD/JPY*NZD/JPY*GBP/CHF"
    #right = "Q/W"
    #right = "Q/W*I/O"
    #req = input("Enter pair to see formula: ")
    #req = "Q/W"
    eq = parseEquation(left, right)
    initValues(eq)
    setInitialEstimates(eq)
    for i in range(1,NUM_ITERATIONS):
        refineEstimates(eq)

    printAll()
