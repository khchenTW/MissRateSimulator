from __future__ import division
from bounds import *
from simulator import MissRateSimulator
import sys
import numpy as np
import sympy as sp
import time
import deadline_miss_probability #this is from ECRTS'18
import decimal
import TDA # this is enhanced from ECRTS'18
import EPST
import task_generator
import mixed_task_builder

# for experiment 5
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

hardTaskFactor = [1.83]

# Setting for Fig4:
#faultRate = [10**-4]
#power = [4]
#utilization = [70]

# Setting for Fig5:
#faultRate = [10**-4]
#power = [4]
#utilization = [75]

# Setting for Fig6:
# faultRate = [10**-2, 10**-4, 10**-6]
# power = [2, 4, 6]
# utilization = [60]

# Setting for plottingS:
faultRate = [10**-4]
power = [4]
utilization = [60]

sumbound = 4
# for the motivational example
#jobnum = 5000000
# for the evalutions
jobnum = 2000000
lookupTable = []
conlookupTable = []

# for task generation
def taskGeneWithTDA(uti, fr, n):
    while (1):
        tasks = task_generator.taskGeneration_p(n,uti)
        #tasks = task_generator.taskGeneration_rounded(n, uti)
        tasks = mixed_task_builder.mixed_task_set(tasks, hardTaskFactor[0], fr)

        if TDA.TDAtest( tasks ) == 0:
            #success, if even normal TDA cannot pass, our estimated missrate is really worse.
            break
        else:
            #fail
            pass
    return tasks

def taskSetInput(n, uti, fr, por, tasksets_amount, part, filename):
    tasksets = [taskGeneWithTDA(uti, fr, n) for x in range(tasksets_amount)]
    np.save(filename, tasksets)
    return filename

def lookup(k, tasks, numDeadline, mode):
    global lookupTable
    global conlookupTable
    if mode == 0:
        if lookupTable[k][numDeadline] == -1:
            lookupTable[k][numDeadline] = EPST.probabilisticTest_k(k, tasks, numDeadline, Chernoff_bounds, 1)
        return lookupTable[k][numDeadline]
    else:
        if conlookupTable[k][numDeadline] == -1:
            #for ECRTS
            probs = []
            states = []
            pruned = []
            conlookupTable[k][numDeadline] = deadline_miss_probability.calculate_pruneCON(tasks, 0.001, probs, states, pruned, numDeadline)
        return conlookupTable[k][numDeadline]

def Approximation(n, J, k, tasks, mode=0):
    # mode 0 == EMR, 1 = CON
    # J is the bound of the idx
    if mode == 0:
        global lookupTable
        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
    else:
        global conlookupTable
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

    probsum = 0
    for x in range(1, J+1):
        probsum += lookup(k, tasks, x, mode)*x

    if probsum == 0:
        return 0
    else:
        #print 1/(1+(1-lookup(k, tasks, 1))/probsum)
        #print probsum/(1+probsum-lookup(k, tasks, 1))
        #for avoiding numerical inconsistance
        return probsum/(1+probsum-lookup(k, tasks, 1, mode))


def experiments_sim(n, por, fr, uti, inputfile):

    SimRateList = []
    ExpectedMissRate = []
    ConMissRate = []
    stampCON = []
    stampEPST = []

    tasksets = np.load(inputfile+'.npy')

    for tasks in tasksets:
        global lookupTable
        global conlookupTable

        simulator=MissRateSimulator(n, tasks)

        # EPST + Theorem2
        # report the miss rate of the lowest priority task

        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        tmp = Approximation(n, sumbound, n-1, tasks, 0)
        if tmp < 10**-4:
            continue
        else:
            ExpectedMissRate.append(tmp)
            ConMissRate.append(Approximation(n, sumbound, n-1, tasks, 1))

        simulator.dispatcher(jobnum, fr)
        SimRateList.append(simulator.missRate(n-1))

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SimRateList
    print "ExpectedMissRate:"
    print ExpectedMissRate
    print "ConMissRate:"
    print ConMissRate


    ofile = "txt/COMPARISON_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("SimRateList:")
    fo.write("\n")
    fo.write(str(SimRateList))
    fo.write("\n")
    fo.write("ConMissRate:")
    fo.write("\n")
    fo.write(str(ConMissRate))
    fo.write("\n")
    fo.write("ExpectedMissRate:")
    fo.write("\n")
    fo.write(str(ExpectedMissRate))
    fo.write("\n")
    fo.close()


def experiments_emr(n, por, fr, uti, inputfile ):
    tasksets = np.load(inputfile+'.npy')
    stampPHIEMR = []
    stampPHICON = []

    ConMissRate = []
    ExpectedMissRate = []
    print "number of tasksets:"+str(len(tasksets))
    for tasks in tasksets:
        ExpectedMissRate.append(Approximation(n, sumbound, n-1, tasks, 0))

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "ExpMissRate:"
    print ExpectedMissRate

    ofile = "txt/EMR_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("Expected Miss Rate:")
    fo.write("\n")
    fo.write(str(ExpectedMissRate))
    fo.write("\n")
    fo.close()

def trendsOfPhiMI(n, por, fr, uti, inputfile):
    tasksets = np.load(inputfile+'.npy')
    ClistRes = []
    listRes = []
    xlistRes = []
    timeEMR = []
    timeCON = []
    for tasks in tasksets:
        CResults = []
        Results = []
        xResults = []
        stampPHIEMR = []
        stampPHICON = []

        for x in range(1, 11):
        #for x in range(1, 4):
            t1 = time.clock()
            r = EPST.probabilisticTest_k(n-1, tasks, x, Chernoff_bounds, 1)
            t2 = time.clock()
            diff = t2-t1
            stampPHIEMR.append(diff)

            Results.append(r)
            xResults.append(r*x)
            if x < 8:
            #if x < 3:
                probs = []
                states = []
                pruned = []
                t3 = time.clock()
                c = deadline_miss_probability.calculate_pruneCON(tasks, 0.001, probs, states, pruned, x)
                t4 = time.clock()
                diff = t4-t3
                stampPHICON.append(diff)
                CResults.append(c)
        timeEMR.append(stampPHIEMR)
        timeCON.append(stampPHICON)
        xlistRes.append(xResults)
        ClistRes.append(CResults)
        listRes.append(Results)

    ofile = "txt/trendsC_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("CPhi")
    fo.write("\n")
    for item in ClistRes:
        fo.write(str(item))
        fo.write("\n")
    fo.write("\n")
    fo.close()

    ofile = "txt/trendsE_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EPhi")
    fo.write("\n")
    for item in listRes:
        fo.write(str(item))
        fo.write("\n")
    fo.write("\n")
    fo.close()

    ofile = "txt/trendsX_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EPhi*j")
    fo.write("\n")
    for item in xlistRes:
        fo.write(str(item))
        fo.write("\n")
    fo.write("\n")
    fo.close()

    ofile = "txt/trendsTIME_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EMR Analysis Time")
    fo.write("\n")
    fo.write("[")
    for item in timeEMR:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")

    fo.write("CON Analysis Time")
    fo.write("\n")
    fo.write("[")
    for item in timeCON:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")

    fo.close()

def experiments_art(n, por, fr, uti, inputfile):
    # this is for artifical test (motivational example)
    SimRateList = []
    tasksets = []
    sets = 20


    tasks = []
    tasks.append({'period': 3, 'abnormal_exe': 2, 'deadline': 3, 'execution': 2, 'prob': 0})
    tasks.append({'period': 5, 'abnormal_exe': 2.25, 'deadline': 5, 'execution': 1, 'prob': 5e-01})

    for x in range(sets):
        tasksets.append(tasks)

    for tasks in tasksets:

        simulator=MissRateSimulator(n, tasks)
        simulator.dispatcher(jobnum, 0.5)
        SimRateList.append(simulator.missRate(n-1))

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SimRateList


    ofile = "txt/ARTresults_task"+str(n)+"_fr"+str(power[por])+"_runs"+str(sets)+".txt"
    fo = open(ofile, "wb")
    fo.write("SimRateList:")
    fo.write("\n")
    fo.write("[")
    for item in SimRateList:
        fo.write(str(item))
        fo.write(",")
    fo.write("]")
    fo.write("\n")
    fo.close()


def ploting_with_s(n, por, fr, uti, inputfile, delta, minS, maxS):
    filePrefix = 'plots-'
    #folder = 'figures/'
    folder = '/home/khchen/Dropbox/' #for working from home

    # assume the inputs are generated
    tasksets = np.load(inputfile+'.npy')
    for idx, tasks in enumerate(tasksets):
        # if idx == 11:
        pp = PdfPages(folder + filePrefix + repr(idx) +'.pdf')

        # sympy Lambdify
        results = []
        hpTasks = tasks[:n-1]
        #print "ScipyNewton:", r1
        for s in np.arange(5, 15, 0.01):
            r1 = EPST.probabilisticTest_s(n-1, tasks, 1, SympyChernoff, s)
            print "ScipyNewton:", r1
            r2 = np.float128()
            r2 = EPST.probabilisticTest_s(n-1, tasks, 1, Chernoff_bounds, s)
            print "EPST:", r2
            if r2 < r1:
                print "EPST is less than r1 when s: ", s
            results.append(r2)
            #print "EPST:"+str(r)
        print "EPST:", min(results)
        '''
        #iteration testing

        for s in np.arange(minS, maxS, delta):
            r = np.float128()
            r = EPST.probabilisticTest_s(n-1, tasks, 1, Chernoff_bounds, s)
            results.append(r)
        title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$:'+repr(uti)+'%' + ', Fault Rate:'+repr(fr) + ', Delta:'+repr(delta)

        plt.title(title, fontsize=20)
        plt.grid(True)
        plt.ylabel('Expected Miss Rate', fontsize=20)
        plt.xlabel('Real number s', fontsize=22)
        plt.yscale("log")
        # ax.set_ylim([10**-28,10**0])
        #ax.tick_params(axis='both', which='major',labelsize=20)
        # labels = ('$10^{-2}$','$10^{-4}$', '$10^{-6}$')
        plt.plot(np.arange(minS, maxS, delta), results, 'ro')
        figure = plt.gcf()
        figure.set_size_inches([10,6.5])

        # plt.legend(handles=[av, box, whisk], fontsize=16, frameon=True, loc=1)

        #plt.show()
        pp.savefig()
        plt.clf()
        pp.close()
        '''



def main():
    args = sys.argv
    if len(args) < 5:
        print "Usage: python experiments.py [mode] [number of tasks] [tasksets_amount] [part]"
        return -1
    # this is used to choose the types of experiments.
    mode = int(args[1])
    n = int(args[2])
    # this is used to generate the number of sets you want to test / load the cooresponding input file.
    tasksets_amount = int(args[3])
    # this is used to identify the experimental sets once the experiments running in parallel.
    part = int(args[4])

    for por, fr in enumerate(faultRate):
        for uti in utilization:
            filename = 'inputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part)
            if mode == 0:
                fileInput=taskSetInput(n, uti, fr, por, tasksets_amount, part, filename)
                print "Generate Task sets:"
                print np.load(fileInput+'.npy')
            elif mode == 1:
                # used to get emr without simulation
                experiments_emr(n, por, fr, uti, filename)
            elif mode == 2:
                # used to get sim results together with emr
                experiments_sim(n, por, fr, uti, filename)
            elif mode == 3:
                # used to get the relationship between phi and phi*i to show the converage.
                trendsOfPhiMI(n, por, fr, uti, filename)
            elif mode == 4:
                # used to present the example illustrating the differences between the miss rate and the probability of deadline misses.
                experiments_art(n, por, fr, uti, filename)
            elif mode == 5:
                # used to print out a continuous curve of results with different real value s
                ploting_with_s(n, por, fr, uti, filename, 0.5, 0, 100)
            else:
                raise NotImplementedError("Error: you use a mode without implementation")

if __name__=="__main__":
    main()
