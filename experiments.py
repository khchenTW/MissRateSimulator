from __future__ import division
from bounds import *
from simulator import MissRateSimulator
import sys
import numpy as np
import timing
#import cprta #this is from SIES'17
import deadline_miss_probability #this is from ECRTS'18
import decimal
import TDA
import EPST
import task_generator
import mixed_task_builder

hardTaskFactor = [1.83]

# Setting for Fig4:
#faultRate = [10**-4]
#power = [4]
#utilization = [70]

# Setting for Fig5:
faultRate = [10**-4]
power = [4]
utilization = [75]

# Setting for Fig6:
#faultRate = [10**-2, 10**-4, 10**-6]
#power = [2, 4, 6]
#utilization = [70]

sumbound = 4
# for the motivational example
#jobnum = 5000000
jobnum = 2000000
lookupTable = []
conlookupTable = []

# for task generation
def taskGeneWithTDA(uti, fr, n):
    while (1):
        tasks = task_generator.taskGeneration_p(n,uti)
        #tasks = task_generator.taskGeneration_rounded(n, uti)
        tasks = mixed_task_builder.mixed_task_set(tasks, hardTaskFactor[0], fr)
        #keepTasks = tasks[:]
        #for i in tasks:
            #print i['period']
            #print i

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
            #for SIES
            #conlookupTable[k][numDeadline] = cprta.cprtao(tasks, numDeadline)
            #for ECRTS
            probs = []
            states = []
            pruned = []
            conlookupTable[k][numDeadline] = deadline_miss_probability.calculate_pruneCON(tasks, 0.001, probs, states, pruned, numDeadline)
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
        # Assume the lowest priority task has maximum...

        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        timing.tlog_start("EPST starts", 1)
        tmp = Approximation(n, sumbound, n-1, tasks, 0)
        timing.tlog_end("EPST finishes", stampEPST, 1)
        if tmp < 10**-4:
            continue
        else:
            ExpectedMissRate.append(tmp)
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
    #just use to quickly get emr
    tasksets = np.load(inputfile+'.npy')
    stampPHIEMR = []
    stampPHICON = []

    ConMissRate = []
    ExpectedMissRate = []
    for tasks in tasksets:
        timing.tlog_start("EMR start", 1)
        ExpectedMissRate.append(Approximation(n, sumbound, n-1, tasks, 0))
        timing.tlog_end("EMR finishes", stampPHIEMR, 1)
        #timing.tlog_start("CON start", 1)
        #ConMissRate.append(Approximation(n, sumbound, n-1, tasks, 1))
        #timing.tlog_end("CON finishes", stampPHICON, 1)

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "ExpMissRate:"
    print ExpectedMissRate
    #print "ConExpMissRate:"
    #print ConMissRate

    ofile = "txt/EMR_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("Expected Miss Rate:")
    fo.write("\n")
    fo.write(str(ExpectedMissRate))
    fo.write("\n")
    #fo.write("Con miss Rate:")
    #fo.write("\n")
    #fo.write(str(ConMissRate))
    #fo.write("\n")
    fo.close()
    ofile = "txt/EMRtime_task"+str(n)+"_fr"+str(power[por])+"_uti"+str(uti)+".txt"
    fo = open(ofile, "wb")
    fo.write("EMR Time:")
    fo.write("\n")
    fo.write(str(stampPHIEMR))
    fo.write("\n")
    #fo.write("CON Time:")
    #fo.write("\n")
    #fo.write(str(stampPHICON))
    #fo.write("\n")
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
            timing.tlog_start("Phi j starts", 1)
            r = EPST.probabilisticTest_k(n-1, tasks, x, Chernoff_bounds, 1)
            timing.tlog_end("Phi j finishes", stampPHIEMR, 1)

            Results.append(r)
            xResults.append(r*x)
            if x < 8:
            #if x < 2:
                probs = []
                states = []
                pruned = []
                timing.tlog_start("Phi CON starts", 1)
                c = deadline_miss_probability.calculate_pruneCON(tasks, 0.001, probs, states, pruned, x)
                timing.tlog_end("Phi CON finishes", stampPHICON, 1)
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

    SimRateList = []
    tasksets = []
    sets = 20

    #this for artifical test

    tasks = []
    tasks.append({'period': 3, 'abnormal_exe': 2, 'deadline': 3, 'execution': 2, 'prob': 0})
    tasks.append({'period': 5, 'abnormal_exe': 2.25, 'deadline': 5, 'execution': 1, 'prob': 5e-01})

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
