# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

import random
import sort_task_set
import math
import numpy
import task


USet=[]
PSet=[]
possiblePeriods = [1, 2, 5, 10, 50, 100, 200, 1000]

def init():
    global USet,PSet
    USet=[]
    PSet=[]

def taskGeneration_rounded( numTasks, uTotal ):
    random.seed()
    init()
    UUniFast_Discard( numTasks, uTotal/100 )
    CSet_generate_rounded( 1, 2 )
    return PSet

def taskGeneration_rounded_random( numTasks, uTotal ):
    random.seed()
    init()
    UUniFast_Discard( numTasks, uTotal/100 )
    CSet_generate_rounded_random_periods( 1, 2 )
    return PSet

def UUniFast_Discard( n, U_avg ):
    while 1:
        sumU = U_avg
        for i in range(1, n):
            #nextSumU = sumU * math.pow( random.random(), 1/(n-i) )
            nextSumU = sumU * numpy.random.random() ** (1.0 / (n - i))
            USet.append( sumU - nextSumU )
            sumU = nextSumU
        USet.append(sumU)
        if max(USet) <= 0.5 and min(USet) > 0.001:
            break
        del USet[:]

def CSet_generate_rounded( Pmin, numLog ):
    global USet,PSet
    while 1:
        executions = []

        #j=0
        for x, i in enumerate(USet):
            #thN=j%numLog
            p = random.randint( 0, len( possiblePeriods) - 1 )  #random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))#calcExecution(Pmin, thN, 10, 2, i)
            period = possiblePeriods[p]                         #round( p, 2 )#*random.uniform(1)
            deadline = period                                   #round( p, 2 )#*random.uniform(1)
            execution = i * period                              #round( i * p, 2 )
            executions.append( execution )
            pair = task.Task( x, period, deadline, execution)
            PSet.append(pair)
            #j=j+1

        #if min(executions) > 0:
        break

        # print("Taskset had 0")
        del PSet[:]
        del executions

def CSet_generate_rounded_random_periods( Pmin, numLog ):
    global USet,PSet
    while 1:
        executions = []

        j=0
        for x, i in enumerate(USet):
            thN=j%numLog
            p = random.uniform(Pmin*math.pow(10, thN), Pmin*math.pow(10, thN+1))#calcExecution(Pmin, thN, 10, 2, i)
            period = round( p, 2 )#*random.uniform(1)
            deadline = round( p, 2 )#*random.uniform(1)
            execution = round( i * p, 2 )
            executions.append( execution )
            pair = task.Task( x, period, deadline, execution)
            PSet.append(pair)
            j=j+1

        #if min(executions) > 0:
        break

        # print("Taskset had 0")
        del PSet[:]
        del executions


def mixed_task_set(tasks, factor):
    allTasks=[]
    for task in tasks:
        task.abnormal_exe = task.execution * factor
        allTasks.append(task)

    return sort_task_set.sort(allTasks, 'period')


# füge zu task ein Prozessor hinzu
def add_processor_to_task( tasks, processorsNum ):
    processors = [0 for x in range(processorsNum)]

    for task in tasks:
        task.uti = task["execution"]/task["period"]

    tasks = sort_task_set.sort(tasks, "uti")
    tasks.reverse()

    for task in tasks:
        processor = lowestUtilizationProcessor(processors)

        uti = task.execution/task.period
        processors[processor] += uti
        task.processor = processor

def lowestUtilizationProcessor(processors):
    x = 0
    minUti = processors[0]
    for i in range(len(processors)):
        if processors[i] < minUti:
            minUti = processors[i]
            x = i
            
    return x

# add a priority to each task
def addPrioritiesToTasks(tasks):
    #print(tasks)

    taskPriorities = [x for x in range(len(tasks))]
    #print(taskPriorities)
    allTasks = []	
    for task in tasks:
        #adds a random priority to a task
        randomPrioIndex = random.random() * len(taskPriorities)
        task.setPriority(taskPriorities.pop(int(randomPrioIndex)) + 1)
        allTasks.append(task)
    return sort_task_set.sort(allTasks, 'priority')


def addPrioritiesToTasksByPeriod(tasks):
    currentPriority = 1
    sortedTasks = sort_task_set.sortEvent(tasks, 'period')
    
    for task in sortedTasks:
        task.setPriority(currentPriority)
        currentPriority += 1
    return sortedTasks

def addPrioritiesToTasksByDeadline(tasks):
    currentPriority = 1
    sortedTasks = sort_task_set.sortEvent(tasks, 'deadline')
    
    for task in sortedTasks:
        task.setPriority(currentPriority)
        currentPriority += 1
    return sortedTasks


def convertArrTasks(arr, processors):
    tasks = []
    periods = [0 for x in range(processors)]
    executions = [0 for x in range(processors)]
    uti = [0.0 for x in range(processors)]
    for a in arr:
        t = task.Task(a['id'], a['period'], a['deadline'], a['execution'])
        t.abnormal_exe = a['abnormal_exe']
        t.priority = a['priority']
        t.processor = a['processor']
        tasks.append(t)
        i = int(a['processor'])
        periods[i] += a['period']
        executions[i] += a['execution']
        uti[i] += a['execution']/a['period']

    #print("Periods: " + str(periods))
    #print("executions: " + str(executions))
    #print("uti: " + str(uti))
    return tasks

def convertArrTasksOrig(arr, processors):
    tasks = []
    uti = 0.0
    for id, a in enumerate(arr):
        t = task.Task(id, a['period'], a['deadline'], a['execution'])
        t.abnormal_exe = a['abnormal_exe']
        tasks.append(t)
        uti += a['execution']/a['period']

    #print("Periods: " + str(periods))
    #print("executions: " + str(executions))
    #print("uti: " + str(uti))
    return tasks


# def main():

#def main():
#	tasks = [{},{},{},{},{},{},{}]
#	print(tasks)
#	addPriorityToTask(tasks)
#	print(tasks)

# if __name__ == "__main__":
#    main()
