from __future__ import division
import math

def Workload_Contrained(T,C,t):
    return C*math.ceil((t)/T)

def TDA(task,HPTasks):
    C=task['execution']
    R=C
    D=task['deadline']

    while True:
        I=0
        for itask in HPTasks:
            I=I+Workload_Contrained(itask['period'], itask['execution'],R)
        if R>D:
            return R
        if R < I+C:
            R=I+C
        else:
            return R

def TDAtest(tasks):
    x = 0
    fail = 0
    for i in tasks:
        hpTasks = tasks[:x]
        RT=TDA(i, hpTasks)
        if RT > i['deadline']:
            fail = 1
            break
        #after this for loop, fail should be 0
        x+=1
    return fail


