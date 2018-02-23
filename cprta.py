from __future__ import division
import random
import math
import time
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
from scipy.optimize import *
from sympy import *
import signal

skipcount = 0

def handler(signo, frame):
    raise RuntimeError

def reportSkipTimes():
    return skipcount

def sortRandVar( var ):
    #sort a random variable in increasing order of its values (first) line)
    return sorted(var, key = lambda x:x[0])
#print sortRandVar([])

def dco(A, B):
    C = B[:]
    for i in A:
        for j in C:
            if i[0] == j[0]:
                j[1]=j[1]+i[1]
                i[0] = -1
    for i in A:
        if i[0] != -1:
            C.append(i)
    return C

def dcfc(A, B):
    #discrete convolution function
    if not A:
        A=[[0,1]]
    if not B:
        B=[[0,1]]
    res = []
#    print "A =", A
#    print "B =", B
    for i in B:
        for j in A:
            tmpC = i[0]+j[0]
            tmpP = i[1]*j[1]
    #        print tmpC, tmpP
            if not res:
                res.append([tmpC, tmpP])
            else:
                flag = -1
                countk = 0
                for k in res:
                    if k[0] == tmpC:
                        flag = countk
                    countk+=1
                if flag == -1:
                    res.append([tmpC, tmpP])
                else:
                    res[flag][1]=res[flag][1]+tmpP
    return sortRandVar(res)
#print dcfc([[3, 0.1],[7, 0.9]],[[0, 0.9],[4, 0.1]])

def cprtao(tasks, numD):
    #special routine for only testing the latest task
    #compute the response time distribution of the last prioritary task in a given taskset in a study interval.
    x = 0
    seqP = []
    #convert tasks dict to list
    plist = []
    clist = []
    alist = []
    c = 0

    for i in tasks:
        clist.append([[i['execution'], 1-i['prob']],[i['abnormal_exe'], i['prob']]])
        plist.append([[i['period'], 1]])
    for i, r in zip(clist, plist): #i['period'], i['execution'], i['abnormal_exe'], i['prob']
        #print "task-",x
        if x != len(clist)-1:
            x+=1
            continue
        hpTasks = clist[:x] #now i is target task and hptasks are all the others
        hpPeriods = plist[:x] #r[0][0] should be the deadline of the lowest priority task
        #print "deadline of the target task"
        #print r[0][0]

        Rtar = i
        for j, k in zip(hpTasks,hpPeriods):
            Rtar = dcfc(Rtar, j)#convolution for C
            alist.append(k)
        #print max(Rtar, key = lambda y:y[0])
        #print min(alist[0], key =lambda y:y[0])
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60*5)
            for j in range(1, int(math.floor(r[0][0]*numD))+1): # here is the testing loop. r[0][0] is the target deadline
                tmpalist = []
                countq = 0
                for k, p in zip(hpTasks, hpPeriods):
                    RtarMC=max(Rtar, key =lambda y:y[0])
                    AjMin=min(alist[countq], key =lambda y:y[0])
                    peri=int(math.floor(AjMin[0]))

                    if RtarMC[0]>AjMin[0] and peri == j: #TDA
                        Rtar = doPreemption(Rtar, alist[countq], k)
                        alist[countq] = dcfc(alist[countq], p) #prob = 1
                    countq+=1
        except RuntimeError:
            global skipcount
            Rtar = [[0,1]]
            skipcount+=1
        Rtar = sortRandVar(Rtar)

        #in Rtar, I expect I will get a response time distribution.
        #r[0][0] is the target deadline
        #for getting the first deadline probability: once the derived item in Rtar that its R is larger than D_target, the probability should be summed.
        tmp = 0.0
        for k in Rtar:
            if k[0] > numD*r[0][0]:
                tmp = tmp + k[1]
        seqP.append(tmp)

    return max(seqP)

def cprta(tasks):
    #compute the response time distribution of the last prioritary task in a given taskset in a study interval.
    x = 0
    seqP = []
    #convert tasks dict to list
    plist = []
    clist = []
    alist = []
    c = 0

    for i in tasks:
        clist.append([[i['execution'], 1-i['prob']],[i['abnormal_exe'], i['prob']]])
        plist.append([[i['period'], 1]])
#    print "c-",clist
#    print "p-",plist
#    print
    count_s=0
    for i, r in zip(clist, plist): #i['period'], i['execution'], i['abnormal_exe'], i['prob']
#        print "task-",count_s
        count_s+=1
        hpTasks = clist[:x] #now i is target task and hptasks are all the others
        hpPeriods = plist[:x]

        Rtar = i
        for j, k in zip(hpTasks,hpPeriods):
            Rtar = dcfc(Rtar, j)#convolution for C
            alist.append(k)
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60*10)
            for j in range(1, int(math.floor(r[0][0]))+1):
                tmpalist = []
                countq = 0
                for k, p in zip(hpTasks, hpPeriods):
                    RtarMC=max(Rtar, key =lambda y:y[0])
                    AjMin=min(alist[countq], key =lambda y:y[0])

                    peri=int(math.floor(AjMin[0]))
                    if RtarMC[0]>AjMin[0] and peri == j: #TDA

                        Rtar = doPreemption(Rtar, alist[countq], k)
                        alist[countq] = dcfc(alist[countq], p) #prob = 1

                    countq+=1
        except RuntimeException:
            global skipcount
            Rtar = [[0, 1]]
            skipcount += 1
        Rtar = sortRandVar(Rtar)


        x+=1
        #find DMP
        tmp = 0.0
        for k in Rtar:
            if k[0] > r[0][0]:
                tmp = tmp + k[1]
        seqP.append(tmp)

    return max(seqP)

def constructHeadTail(randVar, arrival):
    boolean_head_tail = []
    #head is made up of ones, tail is made up of zeros
    for n in randVar:
        if n[0] <= arrival[0]:
            boolean_head_tail.append(1)
        else:
            boolean_head_tail.append(0)
    head = []
    tail = []
    for j, i in zip(boolean_head_tail, randVar):
        if j == 1:
            head = dco(head, [i])
        else:
            tail = dco(tail, [i])
#    print "head-",head
#    print "tail-",tail
    return [head, tail]
#print constructHeadTail([[8, 0.1],[7, 0.9]],[6, 15])


def doPreemption(cRsp, Aj, C):
    #integrates the preemption effect of a job
    #given the response time at the current stage, and the arrival time distribution of a preemption of a preempting job and its execution distribution, it outputs the updated response time
    intermediateR = []
    fake = []
    counti=1
    for i in Aj:
        preemption = Aj[:counti]
        preemptionvalue = []
        preemptionprob = []
        for j in preemption:
            preemptionvalue.append(j[0])
            preemptionprob.append(j[1])
        for j in preemptionprob:
            fake.append([0, j])
        [head, tail] = constructHeadTail(cRsp, preemptionvalue)
        if not tail:
            pass
        else:
            tail = dcfc(tail, C)

        tmp = dco(head, tail) #coalescion
        head = tmp[:]
        resintermediate = dcfc(fake, head)
        for j in resintermediate:
            intermediateR.append(j)
        counti+=1
    return sortRandVar(intermediateR)
#print doPreemption([[5, 0.5], [12, 0.5]],[[10,0.5], [20, 0.5]],[[3, 0.2], [6, 0.8]])







