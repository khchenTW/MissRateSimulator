from __future__ import division
import random
import math
import numpy as np
import mixed_task_builder
import itertools
import task_generator
import operator
import TDA
import EPST
import timing
import cprta
from bounds import *

# for configurations
h = 0
n = 2
sumbound = 4 #old setup for J'
#faultRate = [10**-6, 10**-4, 10**-2]
#faultRate = [10**-4]
#power = [6, 4, 2]
power = [4]
#utilization = [75]
#utilization = [90, 70]
hardTaskFactor = [2.2/1.2]
# for simulator initialization
#lookupTable = [[-1 for x in range(sumbound+3)] for y in range(n)]
lookupTable = [[-1 for x in range(100+3)] for y in range(n)]
conlookupTable = [[-1 for x in range(100+3)] for y in range(n)]
statusTable = [[0 for x in range(4)] for y in range(n)]
eventList = []
tasks = []
stampSIM = []
tmptasks = []

# for task generation
def taskGeneWithTDA(uti, fr):
    while (1):
        tasks = task_generator.taskGeneration_p(n,uti)
        tasks = mixed_task_builder.hardtaskWCET(tasks, hardTaskFactor[0], fr)
        #keepTasks = tasks[:]
        #for i in tasks:
            #print i['period']
            #print i

        if TDA.TDAtest( tasks ) == 0:
            #success, if even normal TDA cannot pass, our estimated missrate is really worse.
            #test(n-1, tasks)
            break
        else:
            #fail
            pass
    return tasks

def taskSetInput(uti, fr, por, tasksets_amount, part):
    tasksets = [taskGeneWithTDA(uti, fr) for n in range(tasksets_amount)]
    np.save('inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part), tasksets)
    return 'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part)

# Example of Periodic Implicit deadline case in the paper (without normal execution)
#tasks = []
#tasks.append({'period': 2, 'abnormal_exe': 1, 'deadline': 2, 'execution': 1, 'type': 'hard', 'prob': 1e-06})
#tasks.append({'period': 5, 'abnormal_exe': 1, 'deadline': 5, 'execution': 3, 'type': 'hard', 'prob': 1e-06})

# Example of Periodic Implicit deadline case in the paper (without normal execution)
#tasks = []
#tasks.append({'period': 2, 'abnormal_exe': 1, 'deadline': 2, 'execution': 1, 'type': 'hard', 'prob': 1e-06})
#tasks.append({'period': 8, 'abnormal_exe': 1, 'deadline': 5, 'execution': 5, 'type': 'hard', 'prob': 1e-06})




"""
per col with 4 rows:
workload
# of release
# of misses
# of deadlines = this should be less than release
"""
def tableReport():
    for i, e in enumerate(eventList):
        print "Event "+str(i)
        print e.case()
        print e.delta
    print
    for x in range(n):
        print "task"+str(x)+": "
        for y in range(4):
            print statusTable[x][y]
    print

class eventClass( object ):
    def __init__(self, case, delta, idx):
        self.eventType = case
        self.delta = delta
        self.idx = idx

    def case(self):
        if self.eventType == 0:
            return "release"
        else:
            return "deadline"

    def updateDelta(self, elapsedTime):
        self.delta = self.delta - elapsedTime

def findTheHighestWithWorkload():
    # if there is no workload in the table, returns -1
    hidx = -1
    for i in range(n):
        if statusTable[i][0] != 0:
            hidx = i
            break
        else:
            pass
    return hidx

def release( idx, fr):
    global h
    global eventList
    # create deadline event to the event list
    #print "Add deadline event for "+str(idx)
    eventList.append(eventClass(1, tmptasks[idx]['deadline'], idx))
    # create release event to the event list
    #print "Add release event for "+str(idx)
    # sporadic randomness
    #spor=tmptasks[idx]['period']+tmptasks[idx]['period']*random.randint(0,20)/100
    #eventList.append(eventClass(0, spor, idx))
    # periodic setup
    eventList.append(eventClass(0, tmptasks[idx]['period'], idx))

    # sort the eventList
    eventList = sorted(eventList, key=operator.attrgetter('delta'))

    # add the workload to the table corresponding entry
    # fault inject

    if random.randint(0,int(1/fr))>int(1/fr)-1:
        statusTable[ idx ][ 0 ] += tmptasks[idx]['abnormal_exe']
    else:
        statusTable[ idx ][ 0 ] += tmptasks[idx]['execution']

    #statusTable[ idx ][ 0 ] += tasks[idx]['execution']

    # decide the highest priority task in the system
    h = findTheHighestWithWorkload()
    if h == -1:
        print "BUG: after release, there must be at least one task with workload."
    statusTable[ idx ][ 1 ]+=1
    #tableReport()

def deadline( idx, fr ):
    # check if the targeted task in the table has workload.
    if workload( idx ) != 0:
        statusTable[ idx ][ 2 ] += 1
    statusTable[ idx ][ 3 ]+=1
    #tableReport()
    # DAC18, if there is no backlog in the lowest priority task, init the simulator again.
    if idx == len(tmptasks)-1 and workload( idx ) == 0:
        eventList = []
        initState(tasks)


def event_to_dispatch( event, fr ):
    # take out the delta from the event
    elapsedTime( event )

    # execute the corresponding event functions
    switcher = {
        0: release,
        1: deadline,
    }
    func = switcher.get( event.eventType, lambda: "ERROR" )
    # execute the event
    func( event.idx, fr )


def elapsedTime( event ):
    global h
    delta = event.delta
    # update the deltas of remaining events in the event list.
    if len(eventList) == 0:
        print "BUG: there is no event in the list to be updated."
    for e in eventList:
        e.updateDelta( delta )
    # update the workloads in the table
    while (delta):
        if h == -1:
            # processor Idle
            delta = 0
        elif delta >= statusTable[h][0]:
            delta = delta - statusTable[h][0]
            statusTable[ h ][ 0 ] = 0
            h = findTheHighestWithWorkload()
        elif delta < statusTable[h][0]:
            statusTable[ h ][ 0 ] -= delta
            delta = 0

def getNextEvent():
    # get the next event from the event list
    event = eventList.pop(0)
    #print "Get Event: "+event.case() + " from " + str(event.idx)
    return event


def missRate(idx):
    # return the miss rate of task idx
    return statusTable[ idx ][ 2 ] / statusTable[ idx ][ 1 ]

def totalMissRate():
    # return the total miss rate of the system
    sumRelease = 0
    sumMisses = 0
    for idx in range(n):
        sumRelease += statusTable[ idx ][ 1 ]
        sumMisses += statusTable[ idx ][ 2 ]
    return sumMisses/sumRelease

def releasedJobs( idx ):
    # return the number of released jobs of idx task in the table
    # print "Released jobs of " + str(idx) + " is " + str(statusTable[ idx ][ 1 ])
    return statusTable[ idx ][ 1 ]

def numDeadlines( idx ):
    # return the number of released jobs of idx task in the table
    # print "Deadlines of " + str(idx) + " is " + str(statusTable[ idx ][ 1 ])
    return statusTable[ idx ][ 3 ]

def releasedMisses( idx ):
    # return the number of misses of idx task in the table
    return statusTable[ idx ][ 2 ]

def workload( idx ):
    # return the remaining workload of idx task in the table
    return statusTable[ idx ][ 0 ]

def initState( tasks ):
    # task release together at 0 without delta / release from the lowest priority task
    tmp=range(len(tasks))
    tmp = tmp[::-1]
    for idx in tmp:
        eventList.append(eventClass(0,0,idx))

def dispatcher( targetedNumber, fr):
    # when the number of released jobs in the lowest priority task is not equal to the targeted number.

    while( targetedNumber != numDeadlines( n - 1 )):
        if len(eventList) == 0:
            print "BUG: there is no event in the dispatcher"
            break
        else:
            event_to_dispatch( getNextEvent(), fr )


def totalAproxMissRate( J ):
    # return the total aprox miss rate of the system
    sumTotal = 0.0
    for x in range(n):
        sumTotal+=Approximation(J, x, tasks)
    return sumTotal/n

def Approximation(J, k, tasks, mode):
    # J is the bound of the idx
    probsum = 0
    for x in range(1, J+1):
        '''
        if probsum == probsum + lookup(k, tasks, x, mode)*x:
            #print "HHHHHH:"+str(x)
            J = x
            break
        else:
            probsum += lookup(k, tasks, x, mode)*x
        '''
        probsum += lookup(k, tasks, x, mode)*x
        #print 'mode: '+str(mode)
        #print 'sum:'
        #print probsum
    if lookup(k, tasks, J, mode)!= 0:
        r = lookup(k, tasks, J+1, mode)/lookup(k, tasks, J, mode)
        if r == 1:
            print "bug: r is not correct"
            return -1
        probsum += lookup(k, tasks, J, mode)/(1-r)

        #print 'Rest mode: '+str(mode)
        #print probsum

    if probsum == 0:
        return 0
    else:
        #print 1/(1+(1-lookup(k, tasks, 1))/probsum)
        #print probsum/(1+probsum-lookup(k, tasks, 1))
        #for avoiding numerical inconsistance
        return probsum/(1+probsum-lookup(k, tasks, 1, mode))

def lookup(k, tasks, numDeadline, mode):
    '''
    # Now is purely for chernoff bounds
    global lookupTable
    #due to array design, numDeadline
    if lookupTable[k][numDeadline] == -1:
        #calcualte
        lookupTable[k][numDeadline] = EPST.probabilisticTest_k(k, tasks, numDeadline, 1)
    return lookupTable[k][numDeadline]
    '''
    global lookupTable
    global conlookupTable
    if mode == 0:
    #due to array design, numDeadline
        if lookupTable[k][numDeadline] == -1:
        #calcualte
            lookupTable[k][numDeadline] = EPST.probabilisticTest_k(k, tasks, numDeadline, Chernoff_bounds, 1)
            #print EPST.probabilisticTest_k(k, tasks, numDeadline, 1)
            #print cprta.cprtao(tasks, numDeadline)
        return lookupTable[k][numDeadline]
    else:
        if conlookupTable[k][numDeadline] == -1:
            conlookupTable[k][numDeadline] = cprta.cprtao(tasks, numDeadline)
            #print EPST.probabilisticTest_k(k, tasks, numDeadline, 1)
            #print cprta.cprtao(tasks, numDeadline)
        return conlookupTable[k][numDeadline]


def test(k, tasks):
    print tasks[k]
    print EPST.probabilisticTest_p(tasks, 1, Chernoff_bounds, 1)
    print EPST.probabilisticTest_k(k, tasks, 1, Chernoff_bounds, 1)
    print EPST.probabilisticTest_k(k, tasks, 2, Chernoff_bounds, 1)
    print EPST.probabilisticTest_k(k, tasks, 3, Chernoff_bounds, 1)
    probsum = 0
    for x in range(1, sumbound+1):
        probsum += EPST.probabilisticTest_k(k, tasks, x, Chernoff_bounds, 1)*x
    print probsum

def experiments_sim(por, fr, uti, inputfile ):

    totalRateList = []
    MaxRateList = []
    ExpectedTotalRate = []
    ExpectedMaxRate = []
    ConMissRate = []

    tasksets = np.load(inputfile+'.npy')

    for tasks in tasksets:
        #print tasks
        global statusTable
        global eventList
        global lookupTable
        global tmptasks
        global conlookupTable
        statusTable = [[0 for x in range(4)] for y in range(n)]
        eventList = []

        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        initState(tasks)
        tmptasks = tasks[:]

        # EPST + Theorem2
        # Assume the lowest priority task has maximum...
        tmp = Approximation(sumbound, n-1, tasks, 0)
        if tmp < 10**-4:
            continue
        else:
            ExpectedMaxRate.append(tmp)
            ConMissRate.append(Approximation(sumbound, n-1, tasks, 1))

        timing.tlog_start("simulator starts", 1)
        dispatcher(2000000, fr)
        MaxRateList.append(missRate(n-1))
        timing.tlog_end("simulator finishes", stampSIM, 1)

        #totalRateList.append(totalMissRate())

        #tableReport()
    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "MaxRateList:"
    print MaxRateList
    print "ExpectedMaxRate:"
    print ExpectedMaxRate
    print "ConMissRate:"
    print ConMissRate


    ofile = "txt/results_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("MaxRateList:")
    fo.write("\n")
    fo.write("[")
    for item in MaxRateList:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")
    fo.write("ConMissRate:")
    fo.write("\n")
    fo.write("[")
    for item in ConMissRate:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")
    fo.write("ExpectedMaxRate:")
    fo.write("\n")
    fo.write("[")
    for item in ExpectedMaxRate:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")
    fo.close()


def experiments_emr(por, fr, uti, inputfile ):
    #just use to quickly get emr
    tasksets = np.load(inputfile+'.npy')

    #ConMissRate = []
    ExpectedMissRate = []
    for tasks in tasksets:
        #print tasks
        #global eventList

        global statusTable
        statusTable = [[0 for x in range(4)] for y in range(n)]

        global lookupTable
        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        #global conlookupTable
        #conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        ExpectedMissRate.append(Approximation(sumbound, n-1, tasks, 0))
        #ConMissRate.append(Approximation(sumbound, n-1, tasks, 1))

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    #print "ConExpMissRate:"
    #print ConMissRate
    print "ExpMissRate:"
    print ExpectedMissRate


def trendsOfPhiMI(por, fr, uti, inputfile):
    tasksets = np.load(inputfile+'.npy')

    IResults = []
    Results = []
    IlistRes = []
    listRes = []
    for tasks in tasksets:
        Results = []
        IResults = []
        for x in range(1, 11):
            r = EPST.probabilisticTest_k(n-1, tasks, x, Chernoff_bounds, 1)
            IResults.append(r*x)
            Results.append(r)
            print r
        IlistRes.append(IResults)
        listRes.append(Results)
    print len(IlistRes)
    print len(listRes)

    ofile = "txt/trendsI_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("Phi*i")
    fo.write("\n")
    for item in IlistRes:
        print item
        fo.write(str(item))
        fo.write("\n")
    fo.close()

    ofile = "txt/trends_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("Phi")
    fo.write("\n")
    for item in listRes:
        print item
        fo.write(str(item))
        fo.write("\n")
    fo.close()

