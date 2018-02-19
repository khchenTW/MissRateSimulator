'''
Author Kevin Huang and Georg von der Brueggen

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

def taskGenerationMatlab(numTasks, uTotal, ind):
    global ofile, PSet
    random.seed()
    init()
    UUniFast(numTasks,uTotal/100)
    CSet_generate(1,2)
    ofile = "taskset-matlab.m"
    PSet = mixed_task_builder.hardtaskWCET(PSet, 1.83, 10**-6.)
    fo = open(ofile, "ab")
    fo.write("if index == "+str(ind)+"\n")
    j = 0
    for i in PSet:
        j+=1
        fo.write("taskSet{"+str(j)+"}{1} = ["+str(i['execution'])+","+str(i['abnormal_exe'])+";0.999999, 0.000001];\n")
        fo.write("taskSet{"+str(j)+"}{2} = ["+str(i['period'])+"; 1];\n")
    fo.write("end\n")
    ofile = "taskset-p.txt"
    fo = open(ofile, "ab")
    print >>fo, json.dumps(PSet)

def taskGenerationMatlabX(numTasks, uTotal, j):
    for i in range(j):
        taskGenerationMatlab(numTasks, uTotal, i)

