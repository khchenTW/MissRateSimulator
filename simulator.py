from __future__ import division
import random
import math
import numpy as np
import operator

# for simulator initialization
class MissRateSimulator:
    def __init__( self, n, tasks ):
        self.statusTable = [[0 for x in range(4)] for y in range(n)]
        self.eventList = []
        self.tasks = tasks
        self.stampSIM = []
        self.h = -1
        self.n = n
        self.initState()
        self.c = 0

        #print("statusTable: " + str(self.statusTable))
        #print("eventList: " + str(self.eventList))
        #print("taks: " + str(self.tasks))
        #print("n: " + str(self.n))


    class eventClass( object ):
        # This is the class of events
        def __init__( self, case, delta, idx ):
            self.eventType = case
            self.delta = delta
            self.idx = idx

        def case(self):
            if self.eventType == 0:
                return "release"
            else:
                return "deadline"

        def updateDelta( self, elapsedTime ):
            self.delta = self.delta - elapsedTime
    """
    The status table for the simulator
    per col with 4 rows:
    workload
    # of release
    # of misses
    # of deadlines = this should be less than release
    """
    def tableReport( self ):
        for i, e in enumerate(self.eventList):
            print("Event "+str(i)+" from task "+str(e.idx))
            print(e.case())
            print(e.delta)

        for x in range(self.n):
            print("task"+str(x)+": ")
            for y in range(4):
                print(self.statusTable[x][y])
            

    def findTheHighestWithWorkload( self ):
        # Assume that the fixed priority is given in the task set.
        # if there is no workload in the table, returns -1
        hidx = -1
        for i in range(self.n):
            if self.statusTable[i][0] != 0:
                hidx = i
                break
            else:
                pass
        return hidx

    def release( self, idx, fr ):
        # create deadline event to the event list
        #print("Add deadline event for "+str(idx))
        #print(idx)
        #print(self.tasks[idx]['deadline'])
        self.eventList.append(self.eventClass(1, self.tasks[idx]['deadline'], idx))
        # create release event to the event list
        #print("Add release event for "+str(idx))
        # sporadic randomness
        #spor=self.tasks[idx]['period']+self.tasks[idx]['period']*random.randint(0,20)/100
        #eventList.append(eventClass(0, spor, idx))
        # periodic setup
        self.eventList.append(self.eventClass(0, self.tasks[idx]['period'], idx))

        # sort the eventList
        self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))

        # add the workload to the table corresponding entry
        # fault inject


        #if random.randint(0,int(1/fr))>int(1/fr)-1:
        if fr == 0:
            self.statusTable[ idx ][ 0 ] += self.tasks[idx]['execution']
        elif fr == 1:
            self.statusTable[ idx ][ 0 ] += self.tasks[idx]['abnormal_exe']
        else:
            #print("com " + str(int(1/fr)-1))
            #print(random.randint(0,int(1/fr)-1))
            #print(int(1/fr)-2)
            if random.randint(0,int(1/fr)-1)>int(1/fr)-2:
                self.statusTable[ idx ][ 0 ] += self.tasks[idx]['abnormal_exe']
            else:
                self.statusTable[ idx ][ 0 ] += self.tasks[idx]['execution']

        # decide the highest priority task in the system
        self.h = self.findTheHighestWithWorkload()
        if self.h == -1:
            print("BUG: after release, there must be at least one task with workload.")
        self.statusTable[ idx ][ 1 ]+=1
        #print("Table in task"+str(idx)+" release event with h"+str(self.h))
        #self.tableReport()

    def deadline( self, idx, fr ):
        # check if the targeted task in the table has workload.
        #print("Table in task"+str(idx)+" deadline event with h"+str(self.h))
        #self.tableReport()
        if self.workload( idx ) != 0:
            print("task"+str(idx)+" misses deadline")
            self.statusTable[ idx ][ 2 ] += 1
        self.statusTable[ idx ][ 3 ]+=1

        #If there is no backlog in the lowest priority task,
        #init the simulator again to force the worst release pattern.
        #TODO this should be done in the release of higher priority task
        if idx == len(self.tasks)-1 and self.workload( idx ) == 0:
            #print("Relase the worst pattern")
            self.eventList = []
            self.c = self.c + 1
            self.initState()



    def event_to_dispatch( self, event, fr ):
        # take out the delta from the event
        self.elapsedTime( event )

        # execute the corresponding event functions
        switcher = {
            0: self.release,
            1: self.deadline,
        }
        func = switcher.get( event.eventType, lambda: "ERROR" )
        # execute the event
        func( event.idx, fr )


    def elapsedTime( self, event ):
        delta = event.delta
        # update the deltas of remaining events in the event list.
        if len(self.eventList) == 0:
            print("BUG: there is no event in the list to be updated.")
        for e in self.eventList:
            e.updateDelta( delta )
        # update the workloads in the table
        while (delta):
            self.h = self.findTheHighestWithWorkload()
            if self.h == -1:
                # processor Idle
                delta = 0
            elif delta >= self.statusTable[self.h][0]:
                delta = delta - self.statusTable[self.h][0]
                self.statusTable[ self.h ][ 0 ] = 0
            elif delta < self.statusTable[self.h][0]:
                self.statusTable[ self.h ][ 0 ] -= delta
                delta = 0

    def getNextEvent(self):
        # get the next event from the event list
        event = self.eventList.pop(0)
        #print("Get Event: "+event.case() + " from " + str(event.idx))
        return event


    def missRate( self, idx):
        # return the miss rate of task idx
        return self.statusTable[ idx ][ 2 ] / self.statusTable[ idx ][ 1 ]

    def totalMissRate( self ):
        # return the total miss rate of the system
        sumRelease = 0
        sumMisses = 0
        for idx in range(n):
            sumRelease += self.statusTable[ idx ][ 1 ]
            sumMisses += self.statusTable[ idx ][ 2 ]
        return sumMisses/sumRelease

    def releasedJobs( self, idx ):
        # return the number of released jobs of idx task in the table
        #print("Released jobs of " + str(idx) + " is " + str(self.statusTable[ idx ][ 1 ]))
        return self.statusTable[ idx ][ 1 ]

    def numDeadlines( self, idx ):
        # return the number of past deadlines of idx task in the table
        #print("Deadlines of " + str(idx) + " is " + str(self.statusTable[ idx ][ 3 ]))
        return self.statusTable[ idx ][ 3 ]

    def releasedMisses( self, idx ):
        # return the number of misses of idx task in the table
        return self.statusTable[ idx ][ 2 ]

    def workload( self, idx ):
        # return the remaining workload of idx task in the table
        return self.statusTable[ idx ][ 0 ]

    def initState( self ):
        # init
        #print(self.tasks)
        self.eventList = []

        # task release together at 0 without delta / release from the lowest priority task
        tmp=range(len(self.tasks))
        tmp = tmp[::-1]
        for idx in tmp:
            self.statusTable[ idx ][ 0 ] = 0
            self.statusTable[ idx ][ 3 ] = self.statusTable[ idx ][ 1 ]
            self.eventList.append(self.eventClass(0,0,idx))
        #self.tableReport()


    def dispatcher(self, targetedNumber, fr):
        # Stop when the number of released jobs in the lowest priority task is equal to the targeted number.

        while( targetedNumber != self.numDeadlines( self.n - 1 )):
            if len(self.eventList) == 0:
                print("BUG: there is no event in the dispatcher")
                break
            else:
                e = self.getNextEvent()
                self.event_to_dispatch(e, fr )
        print("Stop at task "+str(e.idx))
        print(self.c)
        #self.tableReport()




