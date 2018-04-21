'''
Author Kevin Huang, Georg von der Brueggen and Kuan-Hsun Chen
The UUniFast / UUniFast_discard generator.

'''

from __future__ import division
import random
import math
import numpy
import sys, getopt
import json
import mixed_task_builder

ofile = "taskset-p.txt"
USet=[]

def UUniFast(n,U_avg):
    global USet
    sumU=U_avg
    for i in range(n-1):
        nextSumU=sumU*math.pow(random.random(), 1/(n-i))
        USet.append(sumU-nextSumU)
        sumU=nextSumU
    USet.append(sumU)

def UUniFast_Discard(n,U_avg):
    while 1:
        sumU=U_avg
        for i in range(n-1):
            nextSumU=sumU*math.pow(random.random(), 1/(n-i))
            USet.append(sumU-nextSumU)
            sumU=nextSumU
        USet.append(sumU)

        if max(USet) < 1:
                break
        del USet[:]

def UniDist(n,U_min,U_max):
    for i in range(n-1):
        uBkt=random.uniform(U_min, U_max)
        USet.append(uBkt)
"""
def CSet_generate(Pmin,numLog):
	global USet,PSet
	j=0
	for i in USet:
		thN=j%numLog
		#p=random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
		p=random.uniform(10, 100)
		pair={}
		pair['period']=p
		pair['deadline']=p#*random.uniform(1)
		pair['execution']=i*p
		PSet.append(pair)
		j=j+1;
"""
def CSet_generate(Pmin,numLog):
    global USet,PSet
    j=0
    for i in USet:
        thN=j%numLog
        p=random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        pair={}
        pair['period']=p
        pair['deadline']=p#*random.uniform(1)
        pair['execution']=i*p
        PSet.append(pair)
        j=j+1;

def CSet_generate_int(Pmin,numLog):
    global USet,PSet
    j=0
    for i in USet:
        thN=j%numLog
        p=random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        pair={}
        pair['period']=round(p,0)
        pair['deadline']=round(p,0)#*random.uniform(1)
        pair['execution']=round(i*p,0)
        PSet.append(pair)
        j=j+1;

def CSet_generate_rounded(Pmin,numLog):
    global USet,PSet
    j=0
    for i in USet:
        thN=j%numLog
        p=random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))
        pair={}
        pair['period']=round(p,2)
        pair['deadline']=round(p,2)#*random.uniform(1)
        pair['execution']=round(i*p,2)
        PSet.append(pair)
        j=j+1;

def init():
    global USet,PSet
    USet=[]
    PSet=[]

def taskGeneration_p(numTasks,uTotal):
    random.seed()
    init()
    UUniFast(numTasks,uTotal/100)
    CSet_generate(1,2)
    fo = open(ofile, "ab")
    print >>fo, json.dumps(PSet)
    return PSet

def taskGeneration_int(numTasks,uTotal):
    random.seed()
    init()
    UUniFast(numTasks,uTotal/100)
    CSet_generate_int(10,1)
    return PSet

def taskGeneration_rounded(numTasks,uTotal):
    random.seed()
    init()
    UUniFast(numTasks,uTotal/100)
    CSet_generate_rounded(10,2)
    return PSet


