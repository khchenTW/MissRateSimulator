# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

from __future__ import division
from gantt_plotter import GanttPlotter
import random
import math
import numpy as np
import pandas as pd
import operator
import sys
from customEnums import ProcessorType
from customEnums import Scheduling
# from task import Task


# for simulator initialization


class MultiprocessorMissRateSimulator:
    def __init__(self, n, tasks, processors, processorType, scheduling, stopOnMiss):
        self.statusTable = [[0 for i in range(4)] for x in range(n)]
        self.eventList = []
        self.tasks = tasks
        self.stampSIM = []
        self.n = n
        self.processors = processors
        self.processorType = processorType
        self.c = 0
        self.initState()
        self.scheduling = scheduling
        self.stopOnMiss = stopOnMiss
        self.stopSimulator = 0
        self.ganttPlotter = GanttPlotter()
        self.currentTime = 0

        #print("eventList: " + str(self.eventList))
        #print("taks: " + str(self.tasks))
        #print("n: " + str(self.n))
        #print("processor: " + str(self.processor))

    class eventClass(object):
        # This is the class of events
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
    """
    The status table for the simulator
    per col with 4 rows:
    workload
    # of release
    # of misses
    # of deadlines = this should be less than release
    """

    def tableReport(self):
        data = np.array([['','workload','releases', 'misses', 'deadlines']])
        for x in range(self.n):
            newrow = ["Task "+str(x)] + self.statusTable[x]
            data = np.vstack([data, newrow])
        print(pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]))

    def nextScheduledTask(self):
        if self.scheduling == Scheduling.STANDARD.value:
            # standard
            return self.findTheHighestWithWorkload()
        elif self.scheduling == Scheduling.FTP.value:
            # ftp
            return self.fixedTaskPriority()
        elif self.scheduling == Scheduling.EDF.value:
            # edf
            return self.earliestDeadlineFirst()
        elif self.scheduling == Scheduling.DM.value:
            # dm
            return self.fixedTaskPriority()
        elif self.scheduling == Scheduling.RM.value:
            # rm
            return self.fixedTaskPriority()


    def findTheHighestWithWorkload(self):
        # Assume that the fixed priority is given in the task set.
        # if there is no workload in the table, returns -1
        scheduledTasks = [-1] * self.processors
        if self.processorType == ProcessorType.SINGLE.value:
            # single processor simulation
            highestWithWorkloadTaskid = -1
            for task in self.tasks:
                if (not (task.workload() is None)) and (task.workload() > 0):
                    highestWithWorkloadTaskid = task.id
                    return int(highestWithWorkloadTaskid)
                print("BUG: No Tasks given")
        elif self.processorType == ProcessorType.PARTITIONED.value:
            # partitioned
            for processor in range(self.processors):
                highestWithWorkloadTaskid = -1
                for task in self.tasks:
                    if task.processor == processor and not task.workload() is None and task.workload() > 0:
                        highestWithWorkloadTaskid = task.id
                        break
                scheduledTasks[processor] = int(highestWithWorkloadTaskid)
        elif self.processorType == ProcessorType.GLOBAL.value:
            # global
            for processor in range(self.processors):
                highestWithWorkloadTaskid = -1
                for task in self.tasks:
                    if task.id not in scheduledTasks and not task.workload() is None and task.workload() > 0:
                        highestWithWorkloadTaskid = task.id
                        break
                scheduledTasks[processor] = int(highestWithWorkloadTaskid)
        else:
            raise NotImplementedError()
        return scheduledTasks
        
    def fixedTaskPriority(self):
        scheduledTasks = [-1] * self.processors
        highestPrioTaskid = -1
        highestPrio = self.tasks[0].priority
        if self.processorType == ProcessorType.SINGLE.value:
            # single processor simulation
            for task in self.tasks:
                if task.priority <= highestPrio:
                    highestPrio = task.priority
                    highestPrioTaskid = task.id
            if highestPrioTaskid == -1:
                print("BUG: No Tasks given")
            else:
                return highestPrioTaskid
        elif self.processorType == ProcessorType.PARTITIONED.value:
            # partitioned
            for processor in range(0, self.processors):
                highestPrioTaskid = -1
                highestPrio = None
                for task in self.tasks:
                    if task.processor == processor and not task.workload() is None and task.workload() > 0 and (highestPrioTaskid == -1 or (not task.earliestDeadline() is None and task.priority < highestPrio)):
                        highestPrioTaskid = task.id
                        highestPrio = task.priority
                scheduledTasks[processor] = int(highestPrioTaskid)
        elif self.processorType == ProcessorType.GLOBAL.value:
            # global
            for processor in range(0, self.processors):
                highestPrioTaskid = -1
                highestPrio = None
                for task in self.tasks:
                    if task.id not in scheduledTasks and task.workload() > 0 and (highestPrioTaskid == -1 or (task.priority < highestPrio)):
                        highestPrioTaskid = task.id
                        highestPrio = task.priority
                scheduledTasks[processor] = int(highestPrioTaskid)    
        else:
            raise NotImplementedError()
        return scheduledTasks

    def earliestDeadlineFirst(self):
        scheduledTasks = [-1] * self.processors
        earliestDeadlineTaskid = -1
        earliestDeadline = self.tasks[0].earliestDeadline()
        if self.processorType == ProcessorType.SINGLE.value:
            # single processor simulation
            for task in self.tasks:
                if task.earliestDeadline() <= earliestDeadline:
                    earliestDeadline = task.earliestDeadline()
                    earliestDeadlineTaskid = task.id
            if earliestDeadlineTaskid == -1:
                print("BUG: No Tasks given")
            else:
                return earliestDeadlineTaskid
        elif self.processorType == ProcessorType.PARTITIONED.value:
            # partitioned
            for processor in range(0, self.processors):
                earliestDeadlineTaskid = -1
                earliestDeadline = None
                for task in self.tasks:
                    if task.processor == processor and not task.workload() is None and task.workload() > 0 and (earliestDeadlineTaskid == -1 or (not task.earliestDeadline() is None and task.earliestDeadline() < earliestDeadline)):
                        earliestDeadlineTaskid = task.id
                        earliestDeadline = task.earliestDeadline()
                scheduledTasks[processor] = int(earliestDeadlineTaskid)
        elif self.processorType == ProcessorType.GLOBAL.value:
            # global
            for processor in range(0, self.processors):
                earliestDeadlineTaskid = -1
                earliestDeadline = None
                for task in self.tasks:
                    if task.id not in scheduledTasks and task.workload() > 0 and (earliestDeadlineTaskid == -1 or (task.earliestDeadline() < earliestDeadline)):
                        earliestDeadlineTaskid = task.id
                        earliestDeadline = task.earliestDeadline()
                scheduledTasks[processor] = int(earliestDeadlineTaskid)
        else:
            raise NotImplementedError()
        return scheduledTasks


    def release(self, idx, fr):
        tasks = self.getTasksToId( [idx] )

        self.eventList.append(self.eventClass(
            1, tasks[0].deadline + self.currentTime, idx))
        self.eventList.append(self.eventClass(
            0, tasks[0].period + self.currentTime, idx))

        #sort the eventList to ensure correct next event
        self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))

        tasks[0].addJob( fr, self.currentTime )
        self.statusTable[idx][0] += tasks[0].activeJobs[-1].workload

        self.statusTable[idx][1] += 1

    def deadline(self, idx, fr):
        # check if the targeted task in the table has workload.

        tasks = self.getTasksToId( [idx] )

        if len(tasks[0].activeJobs) > 0 and tasks[0].activeJobs[0].deadline <= self.currentTime: 
            self.statusTable[idx][2] += 1

            if self.stopOnMiss:
                self.stopSimulator = 1

        self.statusTable[idx][3] += 1

        # If there is no backlog in the lowest priority task,
        # init the simulator again to force the worst release pattern.
        # TODO this should be done in the release of higher priority task
        lastTask = self.tasks[len(self.tasks) - 1]
        if idx == lastTask.id and lastTask.workload == 0:
            print("Relase the worst pattern " + str(idx) + ": " + str(self.tasks[idx].activeJobs))
            print(str(self.workload(idx)) + ", " + str(self.statusTable[idx][0]) + ", ")
            print("currently ignored, test needs to be better")
            #self.tableReport()
            #self.eventList = []
            self.c = self.c + 1
            #self.initState()

    def event_to_dispatch(self, event, fr):
        # take out the delta from the event
        self.elapsedTime(event)

        # execute the corresponding event functions
        switcher = {
            0: self.release,
            1: self.deadline,
        }

        func = switcher.get(event.eventType, lambda: "ERROR")
        # execute the event
        func(event.idx, fr)

    def elapsedTime(self, event):
        delta = event.delta - self.currentTime

        if len(self.eventList) == 0:
            print("BUG: there is no event in the list to be updated.")

        self.runElapsedTime( delta )

    def runElapsedTime( self, delta ):
        self.currentTime += delta
        while (delta):

            taskIds = self.nextScheduledTask()
            tasks = self.getTasksToId( taskIds )
            tmpDelta = self.findLowestJob( tasks, taskIds, delta )
            
            #self.ganttPlotter.addData(taskIds, tmpDelta)
            for x, taskId in enumerate( taskIds ):
                if not taskId == -1 and tmpDelta == tasks[x].activeJobs[0].workload:
                    #print("removing Job from " + str(taskId) + " with workload " + str(tasks[x].activeJobs[0].workload) + " with tmpDelta: " + str(tmpDelta))
                    tasks[x].activeJobs.pop(0)
                elif not taskId == -1 and tmpDelta < tasks[x].activeJobs[0].workload:
                    #print("removing Workload from Job from " + str(taskId) + " with workload " + str(tasks[x].activeJobs[0].workload))
                    tasks[x].activeJobs[0].workload -= tmpDelta

                #Update Workload in Statustable
                if not taskId == -1:
                    self.statusTable[taskId][0] -= tmpDelta

            delta -= tmpDelta

        if delta < 0:
            print('big fat error**************************')

    def findLowestJob( self, tasks, taskIds, delta ):
        minDelta = delta

        for x, taskId in enumerate( taskIds ): 
            if (not taskId == -1) and (minDelta > tasks[x].activeJobs[0].workload):
                minDelta = tasks[x].activeJobs[0].workload

        return minDelta

    def getTasksToId( self, ids ):
        tasks = []

        for id in ids:
            found = False
            for task in self.tasks:
                if task.id == id:
                    tasks.append(task)
                    found  = True
                    break
            if not found:
                tasks.append(None)

        return tasks

    def updateDeadlines( self, delta):
        for task in self.tasks:
            task.updateDeadline(delta)

    def getNextEvent(self):
        # get the next event from the event list
        event = self.eventList.pop(0)
        return event

    def missRate(self, idx):
        # return the miss rate of task idx
        return self.statusTable[idx][2] / self.statusTable[idx][1]

    def totalMissRate(self):
        # return the total miss rate of the system
        sumRelease = 0
        sumMisses = 0
        for idx in range(self.n):
            sumRelease += self.statusTable[idx][1]
            sumMisses += self.statusTable[idx][2]
        return sumMisses/sumRelease

    def releasedJobs(self, idx):
        # return the number of released jobs of idx task in the table
        return self.statusTable[idx][1]

    def numDeadlines(self, idx):
        # return the number of past deadlines of idx task in the table
        return self.statusTable[idx][3]

    def releasedMisses(self, idx):
        # return the number of misses of idx task in the table
        return self.statusTable[idx][2]

    def workload(self, idx):
        # return the remaining workload of idx task in all processors
        return self.statusTable[idx][0]

    def initState(self):
        # init
        self.eventList = []

        for task in self.tasks:
            task.activeJobs = []
        # task release together at 0 without delta / release from the lowest priority task
        tmp = range(len(self.tasks))
        tmp = tmp[::-1]
        for idx in tmp:
            self.statusTable[idx][0] = 0
            self.statusTable[idx][3] = self.statusTable[idx][1]
            self.eventList.append(self.eventClass(0, 0, idx))

    def dispatcher(self, stopTime, fr):
        # Stop when the time of maxPeriod * jobnum is overstepped or on miss.
        while(self.currentTime <= stopTime and not self.stopSimulator):
            if len(self.eventList) == 0:
                print("BUG: there is no event in the dispatcher")
                break
            else:
                e = self.getNextEvent()
                self.event_to_dispatch(e, fr)

        #check for remaining deadline with potential miss
        e = self.getNextEvent()
        while(e.delta >= self.currentTime):
            # only deadlines, as otherwise new events are added
            if(e.case() == "deadline"):
                self.event_to_dispatch(e, fr)
            if(len(self.eventList) != 0):
                e = self.getNextEvent()
            else:
                break

        # self.ganttPlotter.plot()
        # print("Stop at task " + str(e.idx))
        # print("c:" + str(self.c))
        #self.tableReport()
        #if(not self.hasDeadlineMiss()):
        #    print("delta: " + str(self.allDelta))
        #    print("tmpDelta: " + str(self.allTmpDelta))
        #    print("workedworkload: " + str(self.workedWorkload))
        #    print("workload: " + str(self.allWorkload))
        #    print("workload/processor: " + str(self.allWorkload/self.processors))
        # self.printEvents()
        # self.printTasks()
        #print(self.calculations)

    def hasDeadlineMiss( self ):
        if self.stopOnMiss:
            return self.stopSimulator

        for x in range(len(self.tasks)):
            if self.releasedMisses(x):
                return 1
        return 0

    def printTasks( self ):
        for task in self.tasks:
            print( "task " + str(task.id) + ": " + str(task.activeJobs))

    def printEvents( self ):
        event = self.eventList[0]
        #print( "Event: " + str(event.idx) + ", Type: " + str(event.eventType) + ", Delta: " + str(event.delta))
        for event in self.eventList:
                if event.eventType == 1:
                    print( "Event: " + str(event.idx) + ", Type: " + str(event.eventType) + ", Delta: " + str(event.delta))