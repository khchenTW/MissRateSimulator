from __future__ import division
import random
import math
import time
import numpy as np
import sys, getopt
from scipy.optimize import *
from sympy import *
from bounds import *
maxS = 10
delta = 0.1

def determineWorkload(task, higherPriorityTasks, time):
    # This function is used to accumulate the workload from each task.
    workload = task['abnormal_exe']
    for i in higherPriorityTasks:
        jobs = math.ceil(time / i['period'])
        workload += jobs * i['abnormal_exe']
    return workload

def findpoints(task, higherPriorityTasks, mode = 0):
    points = []
    if mode == 0: #kpoints
        # pick up k testing points here
        for i in higherPriorityTasks:
            point = math.floor(task['deadline']/i['deadline'])*i['deadline']
            if point > 0:
                points.append(point)
        points.append(task['deadline'])
    else: #allpoints
        for i in higherPriorityTasks:
            for r in range(1, int(math.floor(task['period']/i['period']))+1):
                point = r*i['period']
            if point > 0:
                points.append(point)
        points.append(task['deadline'])
    return points

def ktda_list(task, higherPriorityTasks,  ieq, s):
    # This function is used to report a upper bound of the probability for one deadline miss

    kpoints = []
    # pick up k testing points here
    kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    minList = []
    probResList = []
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, t)
        if workload <= t:
            return 0
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)
        if ieq == SympyChernoff:
            # probRes = ieq(task, higherPriorityTasks, fy, s)
            probResList = ieq(task, higherPriorityTasks, fy, s)
        else:
            raise NotImplementedError("Error: You use a bound without implementation.")
        if minP > probResList[0]: #find out the minimum in k points
            minP = probResList[0]
            minT = t
            minList = probResList
    return minList


def ktda_s(task, higherPriorityTasks, ieq, s, mode=0):
    # This function is used to report a upper bound of the probability for one deadline miss


    kpoints = []
    # pick up k testing points here
    kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    selecteds = 0
    minS = 0
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, t)
        if workload <= t:
            return 0
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)
        if ieq == Chernoff_bounds:
            if mode == 0:
                probRes = ieq(task, higherPriorityTasks, fy, s)
            else:
                try:
                    res = minimize_scalar(lambda x : ieq(task, higherPriorityTasks, fy, x), method='bounded', bounds=[0,s])
                    probRes = ieq(task, higherPriorityTasks, fy, res.x)
                    selecteds = res.x
                except TypeError:
                    print "TypeError"
                    probRes = 1
        elif ieq == SympyChernoff:
            ResList = []
            ResList = ieq(task, higherPriorityTasks, fy, s)
            probRes = ResList[0]
            selecteds = ResList[1]
        elif ieq == Hoeffding_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        elif ieq == Bernstein_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        else:
            raise NotImplementedError("Error: You use a bound without implementation.")
        if minP > probRes: #find out the minimum in k points
            minP = probRes
            minS = selecteds
    return [minP, minS]


def ktda_p(task, higherPriorityTasks, ieq, bound):
    # This function is used to report a upper bound of the probability for one deadline miss

    kpoints = []
    # pick up k testing points here
    kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, t)
        if workload <= t:
            return 0
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)
        if ieq == Chernoff_bounds:
            try:
                res = minimize_scalar(lambda x : ieq(task, higherPriorityTasks, fy, x), method='bounded', bounds=[0,bound])
                probRes = ieq(task, higherPriorityTasks, fy, res.x)
            except TypeError:
                print "TypeError"
                probRes = 1
        elif ieq == Hoeffding_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        elif ieq == Bernstein_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        else:
            raise NotImplementedError("Error: You use a bound without implementation.")
        if minP > probRes: #find out the minimum in k points
            minP = probRes
    return minP

def ktda_k(task, higherPriorityTasks,  window, ieq, s):
    # This function is used to report a upper bound of the probability for multiple deadline misses

    kpoints = []
    # pick up k testing points here
    if window != 1:
        for i in higherPriorityTasks:
            for j in range(1, window+1):
                point = math.floor((j)*task['period']/i['period'])*i['period']
                if point != 0.0:
                    kpoints.append(point)
        kpoints.append((window+1)*task['period'])
    else:
        kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    selecteds = 0
    minS = 0
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, t)
        if workload <= t:
            return [0, 0]
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)

        if ieq == Chernoff_bounds:
            try:
                ##find the x with minimum
                res = minimize_scalar(lambda x : ieq(task, higherPriorityTasks, fy, x), method='bounded', bounds=[0,s])
                ##use x to find the minimal
                probRes = ieq(task, higherPriorityTasks, fy, res.x)
            except TypeError:
                print "TypeError"
                probRes = 1
        elif ieq == SympyChernoff:
            ResList = []
            ResList = ieq(task, higherPriorityTasks, fy, s)
            probRes = ResList[0]
            selecteds = ResList[1]
        elif ieq == Hoeffding_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        elif ieq == Bernstein_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        else:
            print ieq
            raise "Error: You use a bound without implementation."

        if minP > probRes: #find out the minimum in k points
            minP = probRes
            minS = selecteds
    return [minP, minS]


def kltda(task, higherPriorityTasks, numDeadline, oneD, ieq, s):
    #oneD is precalculated outside of function call
    tmpList = []
    tmpP = np.float128(0)
    if numDeadline == 0:
        return 1
    elif numDeadline == 1:
        return oneD
    else:
        maxi = np.float128(0)
        for w in range(0, numDeadline):
            tmpList=ktda_k(task, higherPriorityTasks, numDeadline-w, ieq, s)
            tmpP = tmpList[0] * kltda(task, higherPriorityTasks, w, oneD, ieq, s)
            if(tmpP > maxi):
                maxi = tmpP
        return maxi

def probabilisticTest_p(tasks, numDeadline, ieq, bound=1):
    # this function is used to report the maximum deadline miss probability among all the tasks
    seqP = []
    x = 0
    for i in tasks:
        hpTasks = tasks[:x]
        if numDeadline == 1:
            resP = ktda_p(i, hpTasks, ieq, bound)
        else:
            resP = kltda(i, hpTasks, numDeadline, ktda_p(i, hpTasks, ieq, bound),bound)
        seqP.append(resP)
        x+=1
    return max(seqP)

def probabilisticTest_k(k, tasks, numDeadline, ieq, bound=1):
    # this function is used to only test for the task k.
    hpTasks = tasks[:k]
    if numDeadline == 1:
        resP = ktda_p(tasks[k], hpTasks, ieq, bound)
    else:
        resP = kltda(tasks[k], hpTasks, numDeadline, ktda_p(tasks[k], hpTasks, ieq, bound), Chernoff_bounds, bound)
    return resP

def probabilisticTest_s(k, tasks, numDeadline, ieq, s, mode=0):
    # this function is used to only test for task k with different s
    hpTasks = tasks[:k]
    resP = []
    tmpList = []
    if numDeadline == 1:
        tmpList = ktda_s(tasks[k], hpTasks, ieq, s, mode)
        resP = tmpList[0]
    else:
        tmpList = ktda_s(tasks[k], hpTasks, ieq, s, mode)
        resP = kltda(tasks[k], hpTasks, numDeadline, tmpList[0], SympyChernoff, s)
    return resP

def probabilisticTest_list(k, tasks, numDeadline, ieq, s):
    # this function is used to return a list of breakpoint, t, and result.
    hpTasks = tasks[:k]
    resP = []
    if numDeadline == 1:
        resP = ktda_list(tasks[k], hpTasks, ieq, s)
    else:
        print("This should not be called!")
        # resP = kltda(tasks[k], hpTasks, numDeadline, ktda_p(tasks[k], hpTasks, ieq, bound), Chernoff_bounds, bound)
    return resP

