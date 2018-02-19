from __future__ import division
import random
import math
import numpy as np
import sys, getopt
from scipy.optimize import *
from sympy import *

# Assume the input tasks only have two modes: c1 and c2.

def Chernoff_bounds(task, higherPriorityTasks, t, s):
    #t is the tested time t, s is a real number, n is the total number of involved tasks
    '''
    return the upper bounded probability, input the targeted time point t and a real number s
    1. first calculate the total number of jobs among all tasks
    2. calculate mgf function for each task with their corresponding number jobs in nlist
    '''
    prob = 1.0
    #now sumN is the total number of jobs among all the tasks.
    c1, c2, x, p = symbols("c1, c2, x, p")
    expr = exp(c1*x)*(1-p)+exp(c2*x)*p
    mgf = lambdify((c1, c2, x, p), expr)
    #with time ceil(), what's the # of released jobs
    for i in higherPriorityTasks:
        prob = prob * (mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**int(math.ceil(t/i['period']))
    #itself
    prob = prob * (mgf(task['execution'], task['abnormal_exe'], s, task['prob']))**int(math.ceil(t/task['period']))
    prob = prob/exp(s*t)

    return prob

def Hoeffding_inequality(task, higherPriorityTasks, t):
    #t is the tested time t, and n is the total number of involved tasks
    '''
    return the upper bounded probability, input the targeted time point t.
    The detailed implementation can be referred to Theorem 6.
    1. first define two lambdas for the expected value of S_t and (b-a)**2
    2. accumulate them in hep(tau_k)
    '''
    prob = 1.0
    expedSt = 0.0
    sumvar = 0.0
    c1, c2, p = symbols("c1, c2, p")
    sumr = lambdify((c1, c2, p), c1*(1-p)+c2*p)
    # here c1 is ai and c2 is bi
    vari = lambdify((c1, c2), (c2-c1)**2)

    for i in higherPriorityTasks:
        expedSt = expedSt + sumr(i['execution'], i['abnormal_exe'], i['prob'])*int(math.ceil(t/i['period']))
        sumvar = sumvar + vari(i['execution'], i['abnormal_exe'])*int(math.ceil(t/i['period']))
    expedSt = expedSt + sumr(task['execution'], task['abnormal_exe'], task['prob'])*int(math.ceil(t/task['period']))
    sumvar = sumvar + vari(task['execution'], task['abnormal_exe'])*int(math.ceil(t/task['period']))

    if t-expedSt > 0:
        prob = exp(-2*(t-expedSt)**2/sumvar)
    else:
        prob = 1
    return prob


def Bernstein_inequality(task, higherPriorityTasks, t):
    #t is the tested time t, and n is the total number of involved tasks
    '''
    return the upper bounded probability, input the targeted time point t.
    The detailed implementation can be referred to Theorem 8.
    1. define lambda functions for E[C] and E[C**2]
    2. get the corresponding values for K and VarC and E[St]
    '''
    c1, c2, p = symbols("c1, c2, p")
    sumr = lambdify((c1, c2, p), c1*(1-p)+c2*p)
    powerC = lambdify((c1, c2, p), c1*c1*(1-p)+c2*c2*p)

    prob = 1.0
    expedSt = 0.0
    varC = 0.0
    K = 0.0
    tmpC = 0.0
    for i in higherPriorityTasks:
        expedC = sumr(i['execution'], i['abnormal_exe'], i['prob'])
        varC = varC + (powerC(i['execution'], i['abnormal_exe'], i['prob'])-(expedC)**2)*int(math.ceil(t/i['period']))
        expedSt = expedSt + expedC*int(math.ceil(t/i['period']))
        tmpK = max(i['execution']-expedC, i['abnormal_exe']-expedC)
        if tmpK > K:
            K = tmpK
    expedC = sumr(task['execution'], task['abnormal_exe'], task['prob'])
    varC = varC + (powerC(task['execution'], task['abnormal_exe'], task['prob'])-(expedC)**2)*int(math.ceil(t/task['period']))
    expedSt = expedSt + expedC*int(math.ceil(t/task['period']))
    tmpK = max(task['execution']-expedC, task['abnormal_exe']-expedC)
    if tmpK > K:
        K = tmpK
    if t-expedSt > 0:
        prob = exp(-((t-expedSt)**2/2)/(varC+K*(t-expedSt)/3))
    else:
        prob = 1
    return prob
