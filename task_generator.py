# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

'''
Author Kevin Huang, Georg von der Brueggen and Kuan-Hsun Chen
The UUniFast / UUniFast_discard generator.

'''

from __future__ import division
import random
import math
import numpy as np
import sys
import getopt
import json
import mixed_task_builder
import task

ofile = "taskset-p.txt"
USet = []
PSet = []


def UUniFast(n, U_avg):
    global USet
    sumU = U_avg
    for i in range(n-1):
        nextSumU = sumU*math.pow(random.random(), 1/(n-i))
        USet.append(sumU-nextSumU)
        sumU = nextSumU
    USet.append(sumU)


def UUniFast_Discard(n, U_avg):
    while 1:
        sumU = U_avg
        for i in range(n-1):
            nextSumU = sumU*math.pow(random.random(), 1/(n-i))
            USet.append(sumU-nextSumU)
            sumU = nextSumU
        USet.append(sumU)

        if max(USet) < 1:
            break
        del USet[:]


def UUniFastDiscard_Junjie(n, u, nsets):
    sets = []
    while len(sets) < nsets:
        # Classic UUniFast algorithm:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * np.random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)
        # Discard according to specific condition:
        if all((ut <= 0.5 and ut > 0.001) for ut in utilizations):
            sets.append(utilizations)
    return sets


def UniDist(n, U_min, U_max):
    for i in range(n-1):
        uBkt = random.uniform(U_min, U_max)
        USet.append(uBkt)


def CSet_generate(Pmin, numLog):
    global USet, PSet
    j = 0
    for i in USet:
        thN = j % numLog
        p = random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        pair = {}
        pair['period'] = p
        pair['deadline'] = p  # *random.uniform(1)
        pair['execution'] = i*p
        PSet.append(pair)
        j = j+1


def CSet_generate_int(Pmin, numLog):
    global USet, PSet
    j = 0
    for i in USet:
        thN = j % numLog
        p = random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        pair = {}
        pair['period'] = round(p, 0)
        pair['deadline'] = round(p, 0)  # *random.uniform(1)
        pair['execution'] = round(i*p, 0)
        PSet.append(pair)
        j = j+1


def CSet_generate_rounded(Pmin, numLog):
    global USet, PSet
    j = 0
    for x, i in enumerate(USet):
        thN = j % numLog
        # calcExecution(Pmin, thN, 10, 2, i)
        p = random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        period = round( p, 2 )#*random.uniform(1)
        deadline = round( p, 2 )#*random.uniform(1)
        execution = round( i * p, 2 )
        pair = task.Task( x, period, deadline, execution)
        PSet.append(pair)
        j = j+1


def init():
    global USet, PSet
    USet = []
    PSet = []


def taskGeneration_p(numTasks, uTotal):
    random.seed()
    init()
    UUniFast(numTasks, uTotal/100)
    CSet_generate(1, 2)
    return PSet


def taskGeneration_int(numTasks, uTotal):
    random.seed()
    init()
    UUniFast(numTasks, uTotal/100)
    CSet_generate_int(10, 1)
    return PSet


def taskGeneration_rounded(numTasks, uTotal):
    random.seed()
    init()
    UUniFast(numTasks, uTotal/100)
    # CSet_generate_rounded(10,2)
    CSet_generate_rounded(1, 2)
    return PSet

def convertArrTasks(arr):
    tasks = []
    periods = [0.0]
    executions = [0.0]
    uti = [0.0]
    for a in arr:
        t = task.Task(a[0], a[1], a[2], a[3])
        t.abnormal_exe = a[4]
        t.priority = a[5]
        t.processor = a[6]
        t.prob = a[7]
        tasks.append(t)
        i = int(a[6])
        periods[i] += a[1]
        executions[i] += a[3]
        uti[i] += a[3]/a[1]

def calcExecution(Pmin, thN, power, rounding, i):
    # tmp = 0.0
    # while round(tmp, rounding) == 0.0:
    #     tmp = i * (random.uniform(Pmin*math.pow(power, thN), Pmin*math.pow(power, thN+1)))
    #     #print(str(tmp))
    # return tmp
    return i * (random.uniform(Pmin*math.pow(power, thN), Pmin*math.pow(power, thN+1)))
