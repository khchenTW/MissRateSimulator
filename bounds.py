from __future__ import division
from scipy.optimize import bisect
from scipy.optimize import newton
import random
import numpy as np
import sys, getopt
import sympy as sp
import time

flag = 0 # for measure only once

# Assume the input tasks only have two modes: c1 and c2.

def mgf(c1, c2, x, p):
    return str(sp.exp(c1*x)*(1-p)+sp.exp(c2*x)*p)

def SympyChernoff(task, higherPriorityTasks, t, s):
    global flag
    prob = np.float128(1.0)
    np.seterr(all='raise')
    x = sp.symbols("x")

    #version 3
    #combine exp(-st) into each term of the divisor
    #issue: underflow problem in sub-terms

    #expr = 1.0
    #for i in higherPriorityTasks:
    #    expr = sp.Mul(expr, sp.Pow(sp.Mul(sp.exp(sp.Mul(i['execution']-t*sp.ceiling(t/i['period']),x)),(1-i['prob']))+ sp.Mul(sp.exp(sp.Mul(i['abnormal_exe']-t*sp.ceiling(t/i['period']),x)),i['prob']), sp.ceiling(t/i['period'])))
    #expr = sp.Mul(expr, sp.Pow(sp.Mul(sp.exp(sp.Mul(task['execution']-t*sp.ceiling(t/i['period']),x)),(1-task['prob']))+ sp.Mul(sp.exp(sp.Mul(task['abnormal_exe']-t*sp.ceiling(t/i['period']),x)),task['prob']), sp.ceiling(t/task['period'])))
    #mgf1 = sp.lambdify(x, expr)
    #dmgf2 = sp.lambdify(x, expr.diff(x))
    #print expr
    #print mgf1(np.float128(10))

    #version 2 - bisection for first derivative
    expr = 1.0
    expr = expr / sp.exp(x*t)
    for i in higherPriorityTasks:
        expr = sp.Mul(expr, sp.Pow(sp.Mul(sp.exp(sp.Mul(i['execution'],x)),(1-i['prob']))+ sp.Mul(sp.exp(sp.Mul(i['abnormal_exe'],x)),i['prob']), sp.ceiling(t/i['period'])))
    expr = sp.Mul(expr, sp.Pow(sp.Mul(sp.exp(sp.Mul(task['execution'],x)),(1-task['prob']))+ sp.Mul(sp.exp(sp.Mul(task['abnormal_exe'],x)),task['prob']), sp.ceiling(t/task['period'])))

    mgf = sp.lambdify(x, expr)
    dmgf = sp.lambdify(x, expr.diff(x))

    # print "---"
    # print expr
    # print mgf(np.float128(10))
    # print

    # x0 is init guess
    x0 = np.float128(0.0) # dmgf(x0) < 0
    delta = 10
    x1 = np.float128(delta)
    m = np.float128(0)
    eps = np.float128("1e-50")
    while dmgf(x1) < 0:
        # find the upper bound of s
        x1 = x0 + delta
    counter = 0
    while np.float128((x1 - x0)/2) > eps and counter < 50:
        counter += 1
        m = np.float128((x0+x1)/2)
        if dmgf(m) == 0:
            breakpoint = m
            break
        if dmgf(m) > 0:
            x1 = m
        else:
            x0 = m
        '''
        print "x0:", x0
        print "x1:", x1
        print "x1-x0 div 2:", (x1 - x0)/2
        print "dx0:", dmgf(np.float128(x0))
        print "dx1:", dmgf(np.float128(x1))
        print "m:", m
        '''
    # We can also call bisection from the scipy.optimizer:
    # m= bisect(dmgf, np.float128(x0), np.float128(x1))

    start_time = time.time()
    prob = mgf(np.float128(m))
    if flag == 0:
        print ("--- for one t %s seconds ---" % (time.time() - start_time))
        flag = 1

    # newton method from scipy.optimier
    '''
    eps = 1e-5
    x0 = np.float128(0.1)
    x0 = 0.05
    div = expr/expr.diff(x)
    X = newton(mgf, x0, fprime=dmgf, maxiter=100, tol=eps)
    print X
    prob = mgf(np.float128(X))
    print prob
    '''

    # newton method manual implementation
    '''
    counter = 0
    X = x0
    print "init", mgf(X)

    for i in range(1, 100):
        print X
        nextGuess = X - div.subs(x, X)
        X = nextGuess
    if mgf(X) >= 1:
        return np.float128(1.0)
    while sp.Abs(mgf(np.float128(X))) > eps and counter < 200:
        try:
            X = X - np.float128(mgf(X)/dmgf(X))
        except ZeroDivisionError:
            print "Error! - derivative zero for x = ", X
        counter += 1
        print X
        #print mgf(X)
    print "stop"
    print "counter", counter
    prob = mgf(X)
    '''

    '''
    #version 1
    expr = 1.0
    for i in higherPriorityTasks:
        expr = sp.Mul(expr, sp.Pow(sp.exp(i['execution']*x)*(1-i['prob'])+ sp.exp(i['abnormal_exe']*x)*i['prob'], sp.ceiling(t/i['period'])))
    expr = sp.Mul(expr, sp.Pow(sp.exp(task['execution']*x)*(1-task['prob'])+ sp.exp(task['abnormal_exe']*x)*task['prob'], sp.ceiling(t/task['period'])))
    expr = expr / sp.exp(x*t)
    mgf = sp.lambdify(x, expr)
    #mgfprime = expr.diff(x)
    #print mgfprime
    prob = mgf(np.float128(s))
    '''

    '''
    #version 0
    c1, c2, x, p, T = sp.symbols("c1, c2, x, p, T")
    expr = sp.exp(c1*x)*(1-p)+sp.exp(c2*x)*p
    expr = sp.Pow(expr, sp.ceiling(t/T))
    mgf = sp.lambdify((c1, c2, x, p, T), expr)

    for i in higherPriorityTasks:
        prob = prob * mgf(i['execution'], i['abnormal_exe'], s, i['prob'], i['period'])
    prob = prob * mgf(task['execution'], task['abnormal_exe'], s, task['prob'], task['period'])
    prob = prob/sp.exp(s*t)
    '''
    return [prob, m]


def Chernoff_bounds(task, higherPriorityTasks, t, s):
    #t is the tested time t, s is a real number, n is the total number of involved tasks
    '''
    return the upper bounded probability, input the targeted time point t and a real number s
    1. first calculate the total number of jobs among all tasks
    2. calculate mgf function for each task with their corresponding number jobs in nlist
    '''
    count = 0
    prob = 1.0
    probstr = str(prob/sp.exp(s*t))
    b_probstr = str(probstr)
    np.seterr(all='raise')
    # c1, c2, x, p = sp.symbols("c1, c2, x, p")
    # expr = sp.exp(c1*x)*(1-p)+sp.exp(c2*x)*p
    # mgf = sp.lambdify((c1, c2, x, p), expr)
    #with time ceil(), what's the # of released jobs
    for i in higherPriorityTasks:
        count+=1
        # prob = prob * (mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**np.ceil(t/i['period'])
        try:
            b_probstr = str(probstr)
            probstr = str(np.float128(probstr)*np.float128(mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**int(np.ceil(t/i['period'])))
            # if s > 72 and count == 7:
            #     raise Exception
        except Exception as inst:
            print type(inst)
            # print inst
            print "b_prob:"+b_probstr
            print "prob:"+probstr
            print np.float128(str(mgf(i['execution'], i['abnormal_exe'], s, i['prob'])))
            print mgf(i['execution'], i['abnormal_exe'], s, i['prob'])
            print np.ceil(t/i['period'])
            print np.float128(mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**int(np.ceil(t/i['period']))
            print np.float128(b_probstr)*np.float128(mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**int(np.ceil(t/i['period']))
            # print np.float128(mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**np.ceil(t/i['period'])
            # print np.float128(probstr)*np.float128(mgf(i['execution'], i['abnormal_exe'], s, i['prob']))**np.ceil(t/i['period'])
            # print probstr
            # print "taskidx:"+str(count)
    probstr = str(np.float128(probstr) * np.float128(mgf(task['execution'], task['abnormal_exe'], s, task['prob']))**int(np.ceil(t/task['period'])))

    return np.float128(probstr)

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
        expedSt = expedSt + sumr(i['execution'], i['abnormal_exe'], i['prob'])*int(np.ceil(t/i['period']))
        sumvar = sumvar + vari(i['execution'], i['abnormal_exe'])*int(np.ceil(t/i['period']))
    expedSt = expedSt + sumr(task['execution'], task['abnormal_exe'], task['prob'])*int(np.ceil(t/task['period']))
    sumvar = sumvar + vari(task['execution'], task['abnormal_exe'])*int(np.ceil(t/task['period']))

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
        varC = varC + (powerC(i['execution'], i['abnormal_exe'], i['prob'])-(expedC)**2)*int(np.ceil(t/i['period']))
        expedSt = expedSt + expedC*int(np.ceil(t/i['period']))
        tmpK = max(i['execution']-expedC, i['abnormal_exe']-expedC)
        if tmpK > K:
            K = tmpK
    expedC = sumr(task['execution'], task['abnormal_exe'], task['prob'])
    varC = varC + (powerC(task['execution'], task['abnormal_exe'], task['prob'])-(expedC)**2)*int(np.ceil(t/task['period']))
    expedSt = expedSt + expedC*int(np.ceil(t/task['period']))
    tmpK = max(task['execution']-expedC, task['abnormal_exe']-expedC)
    if tmpK > K:
        K = tmpK
    if t-expedSt > 0:
        prob = exp(-((t-expedSt)**2/2)/(varC+K*(t-expedSt)/3))
    else:
        prob = 1
    return prob
