from __future__ import division
import random
import math
import numpy as np
import sys, getopt
from scipy.optimize import *
from sympy import *
from bounds import *
maxS = 10
delta = 0.1

def determineWorkload(task, higherPriorityTasks, criteria, time):
    # This function is used to accumulate the workload from each task.
    workload = task[criteria]
    for i in higherPriorityTasks:
        jobs = math.ceil(time / i['period'])
        workload += jobs * i[criteria]
        #print("jobs " + repr(jobs) + " wl task " + repr(jobs * i[criteria]) + " total workload " + repr(workload))
    return workload

def findpoints(task, higherPriorityTasks, mode = 0):
    points = []
    if mode == 0: #kpoints
        # pick up k testing points here
        for i in higherPriorityTasks:
            point = math.floor(task['period']/i['period'])*i['period']
            if point != 0.0:
                points.append(point)
        points.append(task['period'])
    else: #allpoints
        for i in higherPriorityTasks:
            for r in range(1, int(math.floor(task['period']/i['period']))+1):
                point = r*i['period']
            if point != 0.0:
                points.append(point)
        points.append(task['period'])
    return points

def ktda_s(task, higherPriorityTasks, criteria, ieq, s):
    # This function is used to report a upper bound of the probability for one deadline miss

    kpoints = []
    # pick up k testing points here
    kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, criteria, t)
        if workload <= t:
            return 0
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)
        if ieq == Chernoff_bounds:
            probRes = ieq(task, higherPriorityTasks, fy, s)
        elif ieq == Hoeffding_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        elif ieq == Bernstein_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        else:
            raise NotImplementedError("Error: You use a bound without implementation.")
        if minP > probRes: #find out the minimum in k points
            minP = probRes
    return minP


def ktda_p(task, higherPriorityTasks, criteria, ieq, bound):
    # This function is used to report a upper bound of the probability for one deadline miss

    kpoints = []
    # pick up k testing points here
    kpoints = findpoints(task, higherPriorityTasks, 0)

    # for loop checking k points time
    minP = np.float128(1.0)
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, criteria, t)
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

def ktda_k(task, higherPriorityTasks, criteria, window, ieq, bound):
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
    for t in kpoints:
        workload = determineWorkload(task, higherPriorityTasks, criteria, t)
        if workload <= t:
            return 0
        #as WCET does not pass, check if the probability is acceptable
        fy = float(t)

        if ieq == Chernoff_bounds:
            try:
                ##find the x with minimum
                res = minimize_scalar(lambda x : ieq(task, higherPriorityTasks, fy, x), method='bounded', bounds=[0,bound])
                ##use x to find the minimal
                probRes = ieq(task, higherPriorityTasks, fy, res.x)
            except TypeError:
                print "TypeError"
                probRes = 1
        elif ieq == Hoeffding_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        elif ieq == Bernstein_inequality:
            probRes = ieq(task, higherPriorityTasks, fy)
        else:
            print ieq
            raise "Error: You use a bound without implementation."

        if minP > probRes: #find out the minimum in k points
            minP = probRes
    return minP


def kltda(task, higherPriorityTasks, criteria,  numDeadline, oneD, ieq, bound):
    #oneD is precalculated outside of function call
    if numDeadline == 0:
        return 1
    elif numDeadline == 1:
        return oneD
    else:
        maxi = 0.
        for w in range(0, numDeadline):
            tmpP=ktda_k(task, higherPriorityTasks, criteria,  numDeadline-w, ieq, bound) * kltda(task, higherPriorityTasks, criteria, w, oneD, ieq, bound)
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
            resP = ktda_p(i, hpTasks, 'abnormal_exe', ieq, bound)
        else:
            resP = kltda(i, hpTasks, 'abnormal_exe',  numDeadline, ktda_p(i, hpTasks, 'abnormal_exe', ieq, bound),bound)
        seqP.append(resP)
        x+=1
    return max(seqP)

def probabilisticTest_k(k, tasks, numDeadline, ieq, bound=1):
    # this function is used to only test for the task k.
    hpTasks = tasks[:k]
    if numDeadline == 1:
        resP = ktda_p(tasks[k], hpTasks, 'abnormal_exe', ieq, bound)
    else:
        resP = kltda(tasks[k], hpTasks, 'abnormal_exe',  numDeadline, ktda_p(tasks[k], hpTasks, 'abnormal_exe', ieq, bound), Chernoff_bounds, bound)
    return resP

def probabilisticTest_s(k, tasks, numDeadline, ieq, s):
    # this function is used to only test for task k with different s
    hpTasks = tasks[:k]
    if numDeadline == 1:
        resP = ktda_s(tasks[k], hpTasks, 'abnormal_exe', ieq, s)
    else:
        print("This should not be called!")
        # resP = kltda(tasks[k], hpTasks, 'abnormal_exe',  numDeadline, ktda_p(tasks[k], hpTasks, 'abnormal_exe', ieq, bound), Chernoff_bounds, bound)
    return resP

