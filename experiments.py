# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

from __future__ import division
from multiprocessor_simulator import MultiprocessorMissRateSimulator
import sys
import numpy as np
import time
import decimal
from customEnums import *
import pandas as pd
import os.path

# for experiment 5
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import multiprocessor_task_generator
from _functools import reduce

rcParams.update({'font.size': 15})
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = True

marker = ['o','v','^','s','p','d','o', 'v','+','*','D','x','+','p',]
colors = ['r','g','b','k','m','c','c','m','k','b','r','g','y','m','b']
line = [':',':',':','-','-','-','-','-','-']
line = [':',':',':','-','-','-.','-.','-.','-.','-','-','-','-','-','-']
line = ['-','-','-.',':',':',':','-.','-.','-.','-','-','-','-','-','-']
line = ['-','-','-','--','--','--','-.','-.','-.','-','-','-','-','-','-']


# set allow-pickle equals True, required for saving
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

#scheduling = ProcessorType.SINGLE.value

# Amount of processors
processors = [2, 4, 8]
# Choose the scheduling methods to be executed
schedulingMethods = [2, 3]
# Amount to multiply the execution time by
hardTaskFactor = 1.83
# faultrates
faultRates = [10**-2, 10**-4, 10**-6]
# Utilizations
utilizations = [60, 65, 70, 75, 85, 90, 92, 95, 97, 99, 100] #50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 95, 97, 99, 100
# The types of processor
processorTypes = [1, 2] 
# How often the highest periods task should be executed before stopping
jobnum = 100 

# for task generation
def multiprocessorTaskGeneration(uti, amountTasks, processor, generation):
    
    # Generate Tasks
    if generation == 0:
        tasks = multiprocessor_task_generator.taskGeneration_rounded(amountTasks, uti * processor)
    else:
        tasks = multiprocessor_task_generator.taskGeneration_rounded_random(amountTasks, uti * processor)

    # Add Faulrate and HardTaskFactor
    tasks = multiprocessor_task_generator.mixed_task_set(tasks, hardTaskFactor)

    # Add Processors To Tasks
    multiprocessor_task_generator.add_processor_to_task(tasks, processor)

    # randomly add a priority to each task, can get different priorities later depending on schedule method input
    tasks = multiprocessor_task_generator.addPrioritiesToTasks(tasks)

    # [print(vars(t)) for t in tasks]
    return tasks

def taskSetInput(amountTasks, tasksets_amount, uti, processorAmount, filename, generation):
    tasksets = [np.array(multiprocessorTaskGeneration(uti, amountTasks, processorAmount, generation)) for x in range(tasksets_amount)]
    np.save(filename, tasksets)

def addPriorityByScheduling(tasks, scheduling):
     # if scheduling is FTP (scheduling == Scheduling.FTP.value), randomly add a priority to each task
    if scheduling == Scheduling.FTP.value:
        #print("Adding random priorities to tasks...")
        tasks = multiprocessor_task_generator.addPrioritiesToTasks(tasks)

    # DM scheduling
    elif scheduling == Scheduling.DM.value:
        #print("Adding priorities to tasks by deadline...")
        tasks = multiprocessor_task_generator.addPrioritiesToTasksByDeadline(tasks)

    # RM scheduling
    elif scheduling == Scheduling.RM.value:
        #print("Adding priorities to tasks by period...")
        tasks = multiprocessor_task_generator.addPrioritiesToTasksByPeriod(tasks)
    
    return tasks

def kgV(zahlen):
    mal = 1
    maxVielfaches = 10000000
    vielfaches = 0
    while vielfaches < maxVielfaches:
        vielfaches = zahlen[0] * mal
        try:
            for zahl in zahlen:
                rest = (vielfaches % zahl)
                if not rest: pass
                else:
                    raise
            break
        except: pass
        mal += 1
    if vielfaches >= maxVielfaches:
        #print("kgV Ist größer als " + str(maxVielfaches))
        return jobnum
    return vielfaches

def saveMissrate(missrate, filename, processorAmount, faultrate, taskAmount, taskSetAmount, uti, jobnum):
    
    if os.path.exists(filename + '.npy'):
        missrates = np.load(filename+'.npy')
    else:
        missrates = []

    index = findMissrates(missrates, processorAmount, faultrate, taskAmount, taskSetAmount, uti)

    if not index == -1:
        missrates[index]['missrate'] = missrate
    else:
        temp = {}
        temp['missrate'] = missrate
        temp['processorAmount'] = processorAmount
        temp['faultrate'] = faultrate
        temp['taskAmount'] = taskAmount
        temp['taskSetAmount'] = taskSetAmount
        temp['uti'] = uti
        temp['jobnum'] = jobnum
        missrates = np.append(missrates, temp)
    
    np.save(filename, missrates)

def findMissrates(missrates, processorAmount, faultrate, taskAmount, taskSetAmount, uti):
    for index, missrate in enumerate(missrates):
        if missrate['processorAmount'] == processorAmount and missrate['faultrate'] == faultrate and missrate['taskAmount'] == taskAmount and missrate['taskSetAmount'] == taskSetAmount and missrate['uti'] == uti:
            return index

    return -1

def getMissrate(missrates, processorAmount, faultrate, taskAmount, taskSetAmount, uti):
    for index, missrate in enumerate(missrates):
        if missrate['processorAmount'] == processorAmount and missrate['faultrate'] == faultrate and missrate['taskAmount'] == taskAmount and missrate['taskSetAmount'] == taskSetAmount and missrate['uti'] == uti:
            return missrate['missrate']

    return -1

def printGraph(amountTasksPerProcessor, generation, taskSetAmount, part):
    dataArrays = []
    descriptionArrays = []

    for faultrate in faultRates:
        for processorAmount in processors:
            
            amountTasks = amountTasksPerProcessor * processorAmount
            global_arr = [[uti for uti in utilizations] for s in enumerate(schedulingMethods)]
            partitioned_arr = [[uti for uti in utilizations] for s in enumerate(schedulingMethods)]

            for processorType in processorTypes:
                for y, schedulingMethod in enumerate(schedulingMethods):
                    
                    descriptionArraysTemp = []

                    descriptionArraysTemp.append(reverseMapProcessorType(processorType))
                    descriptionArraysTemp.append(reverseMapScheduling(schedulingMethod))
                    descriptionArraysTemp.append(generation)
                    descriptionArraysTemp.append(processorAmount)
                    descriptionArraysTemp.append(taskSetAmount)
                    descriptionArraysTemp.append(amountTasks)
                    descriptionArraysTemp.append(faultrate)
                    descriptionArraysTemp.append(jobnum)

                    descriptionArrays.append(descriptionArraysTemp)
                    
                    missratesTemp = []
                    for x, uti in enumerate(utilizations):
                        filename = "outputs/"+str(generation)+'_'+str(processorType)+'_'+str(schedulingMethod)
                        if os.path.exists(filename + '.npy'):
                            missrates = np.load(filename+'.npy')
                            missrate = getMissrate(missrates, processorAmount, faultrate, amountTasks, taskSetAmount, uti)
                        else:
                            missrate = -1

                        if missrate == -1:
                            print("The configuration hasn't been simulated yet, simulating now...")
                            print("Configuration: " + "FaultRate: " + str(faultrate) + ", Processor: " + reverseMapProcessorType(processorType) + ", processorAmount: " + str(processorAmount) + ", scheduling: " + reverseMapScheduling(schedulingMethod) + ", UTI: " + str(uti))
                            tasksSimFilename = 'inputs/'+str(amountTasks)+'_'+str(uti)+'_'+str(taskSetAmount)+'_'+str(processorAmount)+'_'+str(generation)+'_'+str(part)
                            test_taskssets_for_misses(amountTasks, faultrate, uti, tasksSimFilename, processorType, processorAmount, schedulingMethod, generation, jobnum)
                            missrates = np.load(filename+'.npy')
                            missrate = getMissrate(missrates, processorAmount, faultrate, amountTasks, taskSetAmount, uti)

                        missratesTemp.append(missrate)

                    dataArrays.append(missratesTemp)
    
    filename = 'outputs/test.pdf'

    cmap = plt.cm.get_cmap("hsv", len(dataArrays) + 1)
    d = {}
    d['x'] = utilizations
    for index, dataArray in enumerate(dataArrays):
        strIndex = descriptionArrays[index][0] + " " + descriptionArrays[index][1]
        d[strIndex] = dataArray

    df=pd.DataFrame(d)
    for index, description in enumerate(descriptionArrays):
        descriptionText = description[0] + " " + description[1]
        plt.plot( 'x', descriptionText, label=descriptionText, marker='o', data=df,  color=cmap(index), linewidth=1)
    plt.ylim(0, 1.0)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(filename)
    plt.close()

    text = ""
    for index, description in enumerate(descriptionArrays):
        text += "\n"
        text += str(index) + ". " + "PType: " + description[0] + ", " + "SMethod: " + description[1] + ", " + "GMethod: " + str(description[2]) + ", " + "PAmount: " + str(description[3]) + ", " + "TSAmount: " + str(description[4]) + ", " + "TAmount: " + str(description[5]) + ", " + "fr: " + str(description[6]) + ", " + "jobnum: " + str(description[7])

    f = open(filename + ".txt", "w")
    f.write(text)
    f.close()

def print_pdf_plot(global_arr, partitioned_arr):
    for faultrate in faultRates:
        for processor in processors:
            amountTasks = 40
            tasksets_amount = 10
            for y, schedulingMethod in enumerate(schedulingMethods):
                filename = 'outputs/'+str(amountTasks)+'_'+str(tasksets_amount)+'_'+str(processor)+'_'+str(schedulingMethod)+'_'+str(jobnum)+'_'+str(faultrate)+'.pdf'
                df=pd.DataFrame({'x': utilizations, 'global': global_arr[y], 'partitioned': partitioned_arr[y]})
                print(str(partitioned_arr) + " " + str(global_arr))
                plt.plot( 'x', 'global', data=df, marker='', color='red', linewidth=2)
                plt.plot( 'x', 'partitioned', data=df, marker='', color='blue', linewidth=2)
                plt.legend()
                plt.savefig(filename) 
                plt.close()

def calculateReleases(filename, processorAmount, taskset_amount, uti):
    
    allReleases = 0
    tasksets = np.load(filename+'.npy')
    tasksets = [multiprocessor_task_generator.convertArrTasks(t, processorAmount) for t in tasksets]
    tasksets_amount = len(tasksets) 

    for idx, tasks in enumerate(tasksets):
        # tasks = sort_task_set.sort(tasks, 'id')

        releases = 0

        maxPeriod = 0
        minPeriod = 10
        for task in tasks:
            if task.period > maxPeriod:
                maxPeriod = task.period
            if task.period < minPeriod:
                minPeriod = task.period

        time = maxPeriod * jobnum
        for task in tasks:
            allReleases += round(time/task['period'])

    averageReleases = allReleases/taskset_amount
    print("processorAmount: " + str(processorAmount) + ", Jobnum: " + str(jobnum) + ", UTI: " + str(uti) + ", averageReleases: " + str(averageReleases) + ", MaxTime: " + str(maxPeriod * jobnum) + ", minPeriod: " + str(minPeriod), ", timeBetween: " + str((maxPeriod * jobnum)/averageReleases))

def main():
    args = sys.argv
    if len(args) < 5:
        print("Usage: python3 experiments.py [mode] [number of tasks] [tasksets_amount] [generationMethod] [part]")
        return -1
    elif len(args) < 7:
        scheduling = Scheduling.STANDARD
    else:
        scheduling = int(args[6])

    # this is used to choose the types of experiments.
    mode = int(args[1])
    # amount of tasks per processor
    amountTasksPerProcessor = int(args[2])
    # this is used to generate the number of sets you want to test / load the cooresponding input file.
    tasksets_amount = int(args[3])
    # this is used to select the generation method 
    generation = int(args[4])
    # this is used to identify the experimental sets once the experiments running in parallel.
    part = int(args[5])


    # Run Simulator with every combination of inputs
    if mode == 3:
        # Make pdf graph
        printGraph(amountTasksPerProcessor, generation, tasksets_amount, part)
        return 0
    
    if mode == 4:
        for processorAmount in processors:
            amountTasks = amountTasksPerProcessor * processorAmount
            for uti in utilizations:
                filename = 'inputs/'+str(amountTasks)+'_'+str(uti)+'_'+str(tasksets_amount)+'_'+str(processorAmount)+'_'+str(generation)+'_'+str(part)
                calculateReleases(filename, processorAmount, tasksets_amount, uti)
        return 0
        

    for faultrate in faultRates:
        for processorAmount in processors:
            
            amountTasks = amountTasksPerProcessor * processorAmount
            global_arr = [[uti for uti in utilizations] for s in enumerate(schedulingMethods)]
            partitioned_arr = [[uti for uti in utilizations] for s in enumerate(schedulingMethods)]

            for x, uti in enumerate(utilizations):
                filename = 'inputs/'+str(amountTasks)+'_'+str(uti)+'_'+str(tasksets_amount)+'_'+str(processorAmount)+'_'+str(generation)+'_'+str(part)

                if mode == 0:
                    print("Generate Task sets: amountTasks: " + str(amountTasks) + ", taskset_amount: " + str(tasksets_amount) + ", uti: " + str(uti) + ", processorAmount: " + str(processorAmount))
                    # Generate Task Set
                    taskSetInput(amountTasks, tasksets_amount, uti, processorAmount, filename, generation)
                    #print( "Tasks after loading from file" )
                    #[print(vars(t)) for t in tasksets[0]]
                elif mode == 2:
                    # Run Simulator
                    for processorType in processorTypes:
                        for y, schedulingMethod in enumerate(schedulingMethods):
                            missrate = test_taskssets_for_misses(amountTasks, faultrate, uti, filename, processorType, processorAmount, schedulingMethod, generation, jobnum)
                            if processorType == 1:
                                partitioned_arr[y][x] = missrate
                            elif processorType == 2:
                                global_arr[y][x] = missrate
                elif mode == 7:
                    # test
                    tasksets_amount = 100
                    n = 10
                    testUtilization = 50
                    processorAmount = 4
                    fr = 10**-4
                    

                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.SINGLE.value, Scheduling.STANDARD.value, processorAmount, part)
                    tests( n, fr, testUtilization, tasksets_amount, ProcessorType.SINGLE.value, Scheduling.FTP.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.SINGLE.value, Scheduling.EDF.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.SINGLE.value, Scheduling.DM.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.SINGLE.value, Scheduling.RM.value, processorAmount, part)


                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.PARTITIONED.value, Scheduling.STANDARD.value, processorAmount, part)
                    tests( n, fr, testUtilization, tasksets_amount, ProcessorType.PARTITIONED.value, Scheduling.FTP.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.PARTITIONED.value, Scheduling.EDF.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.PARTITIONED.value, Scheduling.DM.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.PARTITIONED.value, Scheduling.RM.value, processorAmount, part)

                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.STANDARD.value, processorAmount, part)
                    tests( n, fr, testUtilization, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.FTP.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.EDF.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.DM.value, processorAmount, part)
                    #tests( n, fr, testUtilization, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.RM.value, processorAmount, part)


                    #tests( n, fr, 70, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.FTP.value, processorAmount, part)
                    #tests( n, fr, 70, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.RM.value, processorAmount, part)
                    #tests( n, fr, 70, tasksets_amount, ProcessorType.GLOBAL.value, Scheduling.DM.value, processorAmount, part)
                else:
                    raise NotImplementedError("Error: you use a mode without implementation")

def tests(n, fr, uti, tasksets_amount, processorType, scheduling, processorAmount, part):

    while uti <= 100:
        uti += 5
        filename = 'inputs/'+str(n)+'_'+str(uti)+'_'+str(tasksets_amount)+'_'+ str(reverseMapProcessorType(processorType)) + '_' + str(part)
        taskSetInput(n, tasksets_amount, uti, processorAmount, filename)
        test_taskssets_for_misses(n, fr, uti, filename, processorType, processorAmount, scheduling)
    print(" ")

def test_taskssets_for_misses(n, fr, uti, inputfile, processorType, processorAmount, scheduling, generation, jobnum):

    c = 0

    tasksets = np.load(inputfile+'.npy')
    tasksets = [multiprocessor_task_generator.convertArrTasks(t, processorAmount) for t in tasksets]
    tasksets_amount = len(tasksets) 

    for idx, tasks in enumerate(tasksets):
        maxPeriod = 0
        periods = []
        for task in tasks:
            periods.append(task.period)
            if task.period > maxPeriod:
                maxPeriod = task.period

        if generation == 0:
            kgVPeriods = kgV(periods)
            jobnum = kgVPeriods/maxPeriod

        tasks = addPriorityByScheduling(tasks, scheduling)

        if processorType == ProcessorType.SINGLE.value:
            simulator=MultiprocessorMissRateSimulator(n, tasks, 1, 1, scheduling, 1)
        elif processorType == ProcessorType.PARTITIONED.value or processorType == ProcessorType.GLOBAL.value:
            simulator=MultiprocessorMissRateSimulator(n, tasks, processorAmount, processorType, scheduling, 1)
        else:
            raise ValueError()

        simulator.dispatcher(maxPeriod * jobnum, fr)

        if simulator.hasDeadlineMiss():
            c += 1
            #print("missed " + str(idx))
        #else:
        #    print("not missed")
        
    print("FaultRate: " + str(fr) + ", Processor: " + reverseMapProcessorType(processorType) + ", processorAmount: " + str(processorAmount) + ", scheduling: " + reverseMapScheduling(scheduling) + ", UTI: " + str(uti) + ", missed atleast once: " + str(c) + " out of " + str(tasksets_amount))
    saveFilename = "outputs/"+str(generation)+'_'+str(processorType)+'_'+str(scheduling)
    saveMissrate(c/tasksets_amount, saveFilename, processorAmount, fr, len(tasks), tasksets_amount, uti, jobnum)
    return c/tasksets_amount

if __name__=="__main__":
    main()
