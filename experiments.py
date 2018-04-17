from __future__ import division
from bounds import *
from simulator import MissRateSimulator
import sys
import numpy as np
import timing
import cprta
import decimal
import TDA
import EPST
import task_generator
import mixed_task_builder

faultRate = [10**-4]
#faultrate must be in the range between 0 and 1
hardTaskFactor = [2.2/1.2]
# this list is used to generate a readible name of output.
power = [4]
utilization = [75]
sumbound = 4
jobnum = 5000000
lookupTable = []
conlookupTable = []
'''
lookupTable = [[-1 for x in range(sumbound+3)] for y in range(n)]
conlookupTable = [[-1 for x in range(sumbound+3)] for y in range(n)]
'''

# for task generation
def taskGeneWithTDA(uti, fr, n):
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

def taskSetInput(n, uti, fr, por, tasksets_amount, part, filename):
    tasksets = [taskGeneWithTDA(uti, fr, n) for x in range(tasksets_amount)]
    np.save(filename, tasksets)
    return filename

# Example of Periodic Implicit deadline case in the paper (without normal execution)
#tasks = []
#tasks.append({'period': 2, 'abnormal_exe': 1, 'deadline': 2, 'execution': 1, 'type': 'hard', 'prob': 1e-06})
#tasks.append({'period': 5, 'abnormal_exe': 1, 'deadline': 5, 'execution': 3, 'type': 'hard', 'prob': 1e-06})

# Example of Periodic Implicit deadline case in the paper (without normal execution)
#tasks = []
#tasks.append({'period': 2, 'abnormal_exe': 1, 'deadline': 2, 'execution': 1, 'type': 'hard', 'prob': 1e-06})
#tasks.append({'period': 8, 'abnormal_exe': 1, 'deadline': 5, 'execution': 5, 'type': 'hard', 'prob': 1e-06})

def lookup(k, tasks, numDeadline, mode):
    global lookupTable
    global conlookupTable
    if mode == 0:
        if lookupTable[k][numDeadline] == -1:
            lookupTable[k][numDeadline] = EPST.probabilisticTest_k(k, tasks, numDeadline, Chernoff_bounds, 1)
        return lookupTable[k][numDeadline]
    else:
        if conlookupTable[k][numDeadline] == -1:
            conlookupTable[k][numDeadline] = cprta.cprtao(tasks, numDeadline)
        return conlookupTable[k][numDeadline]

def Approximation(n, J, k, tasks, mode=0):
    # mode 0 == EPST, 1 = CPRTA
    # J is the bound of the idx
    if mode == 0:
        global lookupTable
        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
    else:
        global conlookupTable
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

    #print 'Approximation mode: '+str(mode)
    probsum = 0
    for x in range(1, J+1):
        probsum += lookup(k, tasks, x, mode)*x
        #print 'precise part:'
        #print probsum
    '''
    if lookup(k, tasks, J, mode)!= 0:
        if lookup(k, tasks, J+1, mode) != lookup(k, tasks, J, mode):
            r = decimal.Decimal(lookup(k, tasks, J+1, mode)/lookup(k, tasks, J, mode))
            print "r="+str(decimal.Decimal(r))
        else:
            print "bug: r is not correct"
            return -1
        probsum += decimal.Deciaml(lookup(k, tasks, J, mode)/(1-r))
        print 'approximation part:'
        print probsum
    '''

    if probsum == 0:
        return 0
    else:
        #print 1/(1+(1-lookup(k, tasks, 1))/probsum)
        #print probsum/(1+probsum-lookup(k, tasks, 1))
        #for avoiding numerical inconsistance
        return probsum/(1+probsum-lookup(k, tasks, 1, mode))


def experiments_sim(n, por, fr, uti, inputfile):

    totalRateList = []
    SimRateList = []
    ExpectedTotalRate = []
    ExpectedMaxRate = []
    ConMissRate = []
    stampCON = []
    stampEPST = []

    tasksets = np.load(inputfile+'.npy')

    for tasks in tasksets:
        global lookupTable
        global conlookupTable

        simulator=MissRateSimulator(n, tasks)

        # EPST + Theorem2
        # Assume the lowest priority task has maximum...

        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        timing.tlog_start("EPST starts", 1)
        tmp = Approximation(n, sumbound, n-1, tasks, 0)
        timing.tlog_end("EPST finishes", stampEPST, 1)
        if tmp < 10**-4:
            continue
        else:
            ExpectedMaxRate.append(tmp)
            timing.tlog_start("convolution starts", 1)
            ConMissRate.append(Approximation(n, sumbound, n-1, tasks, 1))
            timing.tlog_end("convolution finishes", stampCON, 1)


        timing.tlog_start("simulator starts", 1)
        simulator.dispatcher(jobnum, fr)
        SimRateList.append(simulator.missRate(n-1))
        timing.tlog_end("simulator finishes", simulator.stampSIM, 1)

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SimRateList
    print "ExpectedMaxRate:"
    print ExpectedMaxRate
    print "ConMissRate:"
    print ConMissRate


    ofile = "txt/results_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("SimRateList:")
    fo.write("\n")
    fo.write("[")
    for item in SimRateList:
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


def experiments_emr(n, por, fr, uti, inputfile ):
    #just use to quickly get emr
    tasksets = np.load(inputfile+'.npy')
    stampPHIEMR = []
    stampPHICON = []

    ConMissRate = []
    ExpectedMissRate = []
    for tasks in tasksets:
    #tasks = tasksets[0]
        timing.tlog_start("EMR start", 1)
        ExpectedMissRate.append(Approximation(n, sumbound, n-1, tasks, 0))
        timing.tlog_end("EMR finishes", stampPHIEMR, 1)
        timing.tlog_start("CON start", 1)
        ConMissRate.append(Approximation(n, sumbound, n-1, tasks, 1))
        timing.tlog_end("CON finishes", stampPHICON, 1)

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "ExpMissRate:"
    print ExpectedMissRate
    print "ConExpMissRate:"
    print ConMissRate


def trendsOfPhiMI(n, por, fr, uti, inputfile):
    tasksets = np.load(inputfile+'.npy')

    ExpectedMissRate = []
    stampPHIEMR = []
    stampPHICON = []
    CResults = []
    Results = []
    xRestuls = []
    ClistRes = []
    listRes = []
    xlistRes = []
    for tasks in tasksets:
        Results = []
        IResults = []
        for x in range(1, 11):
            timing.tlog_start("Phi EMR starts", 1)
            r = EPST.probabilisticTest_k(n-1, tasks, x, Chernoff_bounds, 1)
            timing.tlog_end("Phi EMR finishes", stampPHIEMR, 1)
            timing.tlog_start("EMR start", 1)
            ExpectedMissRate.append(Approximation(n, sumbound, n-1, tasks, 0))
            timing.tlog_end("EMR finishes", stampPHIEMR, 1)

            #timing.tlog_start("Phi CON starts", 1)
            #c = cprta.cprtao(tasks, x)
            #timing.tlog_end("Phi CON finishes", stampPHICON, 1)
            Results.append(r)
            #CResults.append(c)
            xRestuls.append(r*x)
        xlistRes.append(xResults)
        #ClistRes.append(CResults)
        listRes.append(Results)

    ofile = "txt/trendsC_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("CPhi")
    fo.write("\n")
    for item in ClistRes:
        print item
        fo.write(str(item))
        fo.write("\n")
    fo.close()

    ofile = "txt/trendsE_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EPhi")
    fo.write("\n")
    for item in listRes:
        print item
        fo.write(str(item))
        fo.write("\n")
    fo.close()

    ofile = "txt/trendsX_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EPhi*j")
    fo.write("\n")
    for item in xlistRes:
        print item
        fo.write(str(item))
        fo.write("\n")
    fo.close()

def experiments_art(n, por, fr, uti, inputfile):

    SimRateList = []
    tasksets = []
    sets = 20

    #this for artifical test

    tasks = []
    tasks.append({'period': 3, 'abnormal_exe': 2, 'deadline': 3, 'execution': 2, 'type': 'hard', 'prob': 0})
    tasks.append({'period': 5, 'abnormal_exe': 2.25, 'deadline': 5, 'execution': 1, 'type': 'hard', 'prob': 5e-01})

    for x in range(sets):
        tasksets.append(tasks)

    for tasks in tasksets:

        simulator=MissRateSimulator(n, tasks)

        timing.tlog_start("simulator starts", 1)
        simulator.dispatcher(jobnum, 0.5)
        SimRateList.append(simulator.missRate(n-1))
        timing.tlog_end("simulator finishes", simulator.stampSIM, 1)

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
                #use to quickly get emr without simulation
                experiments_emr(n, por, fr, uti, filename)
            elif mode == 2:
                #use to get sim results together with emr
                experiments_sim(n, por, fr, uti, filename)
            elif mode == 3:
                #use to get the relationship between phi and phi*i to show the converage.
                trendsOfPhiMI(n, por, fr, uti, filename)
            elif mode == 4:
                #use  to present the example illustrating the differences between the miss rate and the probability of deadline misses.
                experiments_art(n, por, fr, uti, filename)
            else:
                raise NotImplementedError("Error: you use a mode without implementation")

if __name__=="__main__":
    main()
