from __future__ import division
import random
import math
import numpy as np
import operator

# for configurations
h = 0
n = 2
# for simulator initialization
statusTable = [[0 for x in range(4)] for y in range(n)]
eventList = []
tasks = []
stampSIM = []
tmptasks = []

"""
The status table for the simulator
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
    # Stop when the number of released jobs in the lowest priority task is not equal to the targeted number.

    while( targetedNumber != numDeadlines( n - 1 )):
        if len(eventList) == 0:
            print "BUG: there is no event in the dispatcher"
            break
        else:
            event_to_dispatch( getNextEvent(), fr )





