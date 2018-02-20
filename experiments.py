from __future__ import division
from bounds import *
from simulator import MissRateSimulator
import sys
import numpy as np
import timing
import cprta
import TDA
import EPST
import task_generator
import mixed_task_builder


faultRate = [10**-4]
#faultrate must be in the range between 0 and 1
hardTaskFactor = [2.2/1.2]
n = 2
# this list is used to generate a readible name of output.
power = [4]
utilization = [75]
sumbound = 4
jobnum = 100000
lookupTable = [[-1 for x in range(sumbound+3)] for y in range(n)]
conlookupTable = [[-1 for x in range(sumbound+3)] for y in range(n)]

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


def totalAproxMissRate( J ):
    # return the total aprox miss rate of the system
    sumTotal = 0.0
    for x in range(n):
        sumTotal+=Approximation(J, x, tasks)
    return sumTotal/n

def experiments_sim(por, fr, uti, inputfile):

    totalRateList = []
    MaxRateList = []
    ExpectedTotalRate = []
    ExpectedMaxRate = []
    ConMissRate = []

    tasksets = np.load(inputfile+'.npy')

    for tasks in tasksets:
        global lookupTable
        global conlookupTable

        simulator=MissRateSimulator(n, tasks)

        # EPST + Theorem2
        # Assume the lowest priority task has maximum...

        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        tmp = Approximation(sumbound, n-1, tasks, 0)
        if tmp < 10**-4:
            continue
        else:
            ExpectedMaxRate.append(tmp)
            ConMissRate.append(Approximation(sumbound, n-1, tasks, 1))


        timing.tlog_start("simulator starts", 1)
        simulator.dispatcher(jobnum, fr)
        MaxRateList.append(simulator.missRate(n-1))
        timing.tlog_end("simulator finishes", simulator.stampSIM, 1)

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

        #global statusTable
        #statusTable = [[0 for x in range(4)] for y in range(n)]

        global lookupTable
        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        #global conlookupTable
        #conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        ExpectedMissRate.append(Approximation(sumbound, n-1, tasks, 0))
        #ConMissRate.append(Approximation(sumbound, n-1, tasks, 1))

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "ExpMissRate:"
    print ExpectedMissRate
    #print "ConExpMissRate:"
    #print ConMissRate


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



def main():
    args = sys.argv
    if len(args) < 4:
        print "Usage: python experiments.py [mode] [tasksets_amount] [part]"
        return -1
    # this is used to choose the types of experiments.
    mode = int(args[1])
    # this is used to generate the number of sets you want to test / load the cooresponding input file.
    tasksets_amount = int(args[2])
    # this is used to identify the experimental sets once the experiments running in parallel.
    part = int(args[3])
    if mode == 0:
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                fileInput=taskSetInput(uti, fr, por, tasksets_amount, part)
                print "Generate Task sets:"
                print np.load(fileInput+'.npy')
    elif mode == 1:
        #use to quickly get emr without simulation
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                experiments_emr(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))
    elif mode == 2:
        #use to get sim results together with emr
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                experiments_sim(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))
    elif mode == 3:
        #use to get the relationship between phi and phi*i to show the converage.
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                trendsOfPhiMI(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))

if __name__=="__main__":
    main()
