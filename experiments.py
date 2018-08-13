from __future__ import division
from bounds import *
from simulator import MissRateSimulator
import sys
import numpy as np
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
faultRate = [10**-4]
power = [4]
utilization = [75]

# Setting for Fig6:
# faultRate = [10**-2, 10**-4, 10**-6]
# power = [2, 4, 6]
# utilization = [60]

# Setting for plottingS:
#faultRate = [10**-4]
#power = [4]
#utilization = [60]

sumbound = 4
# for the motivational example
#jobnum = 5000000
# for the evalutions
jobnum = 2000000
#jobnum = 2
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

def lookup(k, tasks, fr, numDeadline, mode):
    global lookupTable
    global conlookupTable
    if mode == 0:
        if lookupTable[k][numDeadline] == -1:
            lookupTable[k][numDeadline] = EPST.probabilisticTest_s(k, tasks, numDeadline, SympyChernoff, -1)
        return lookupTable[k][numDeadline]
    else:
        if conlookupTable[k][numDeadline] == -1:
            #for ECRTS
            probs = []
            states = []
            pruned = []
            conlookupTable[k][numDeadline] = deadline_miss_probability.calculate_pruneCON(tasks, fr, probs, states, pruned, numDeadline)
        return conlookupTable[k][numDeadline]

def Approximation(n, fr, J, k, tasks, mode=0):
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
        probsum += lookup(k, tasks, fr, x, mode)*x

    if probsum == 0:
        return 0
    else:
        #print 1/(1+(1-lookup(k, tasks, 1))/probsum)
        #print probsum/(1+probsum-lookup(k, tasks, 1))
        #for avoiding numerical inconsistance
        return probsum/(1+probsum-lookup(k, tasks, fr, 1, mode))


def experiments_sim(n, por, fr, uti, inputfile):

    Outputs = True
    filePrefix = 'sim'
    folder = 'figures/'
    pp = PdfPages(folder + "task" + repr(n) + "-" + filePrefix + '.pdf')
    SimRateList = []
    ExpectedMissRate = []
    ConMissRate = []
    stampCON = []
    stampEPST = []

    tasksets = np.load(inputfile+'.npy')
    tasksets_amount = len(tasksets)
    pass_amount = 0
    try:
        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
        SIM = np.load(filename+'.npy')
        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        EMR = np.load(filename+'.npy')
        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        CON = np.load(filename+'.npy')
    except IOError:
        Outputs = False

    if Outputs is False:
        for idx, tasks in enumerate(tasksets):

            if pass_amount == 20:
                break
            global lookupTable
            global conlookupTable

            print "Running simulator"
            simulator=MissRateSimulator(n, tasks)

            # EPST + Theorem2
            # report the miss rate of the lowest priority task

            lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
            conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

            print "Approximate the miss rate"
            tmp = Approximation(n, fr, sumbound, n-1, tasks, 0)
            if tmp < 10**-4:
                continue
            else:
                pass_amount += 1
                ExpectedMissRate.append(tmp)
                ConMissRate.append(Approximation(n, fr, sumbound, n-1, tasks, 1))

            simulator.dispatcher(jobnum, fr)
            SimRateList.append(simulator.missRate(n-1))

        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
        np.save(filename, SimRateList)
        SIM = np.load(filename+'.npy')
        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, ExpectedMissRate)
        EMR = np.load(filename+'.npy')
        filename = 'outputs/'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, ConMissRate)
        CON = np.load(filename+'.npy')
    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SIM
    print "ExpectedMissRate:"
    print EMR
    print "ConMissRate:"
    print CON


    '''
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
    '''
    #prune for leq 20 sets
    print "Num of SIM:",len(SIM)
    print "Num of CON:",len(CON)
    print "Num of EMR:",len(EMR)
    if len(SIM) > 20:
        SIM = SIM[:20]
        EMR = EMR[:20]
        CON = CON[:20]

    width = 0.15
    ind = np.arange(20) # the x locations for the groups
    title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$:'+repr(uti)+'%' + ', Fault Rate:'+repr(fr)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.ylabel('Expected Miss Rate', fontsize=20)
    plt.yscale("log")
    plt.ylim([10**-5, 10**0])
    pltlabels = []
    for idt, tt in enumerate(SIM):
        pltlabels.append('S'+str(idt))
    plt.xticks(ind + width /2, pltlabels)
    plt.tick_params(axis='both', which='major',labelsize=18)
    print ind
    try:
        rects1 = plt.bar(ind-0.1, SIM, width, color='black', edgecolor='black')
        rects2 = plt.bar(ind+0.1, CON, width, fill = False, edgecolor='black')
        rects3 = plt.bar(ind+0.3, EMR, width, edgecolor='black', hatch='/')
        plt.legend((rects1[0], rects2[0], rects3[0]),('SIM', 'CON', 'EMR'))
    except ValueError:
        print "Value ERROR!!!!!!!!!!"
    figure = plt.gcf()
    figure.set_size_inches([14.5,6.5])

    # plt.legend(handles=[av, box, whisk], fontsize=16, frameon=True, loc=1)

    #plt.show()
    pp.savefig()
    plt.clf()
    pp.close()



def experiments_emr(n, por, fr, uti, inputfile ):

    tasksets = np.load(inputfile+'.npy')
    stampPHIEMR = []
    stampPHICON = []

    ConMissRate = []
    ExpectedMissRate = []
    print "number of tasksets:"+str(len(tasksets))
    for tasks in tasksets:
        global lookupTable
        global conlookupTable
        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

        ExpectedMissRate.append(Approximation(n, fr, sumbound, n-1, tasks, 0))

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

'''
def ploting_with_s(n, por, fr, uti, inputfile, delta, minS, maxS):
    filePrefix = 'plots-'
    folder = 'figures/'

    # assume the inputs are generated
    tasksets = np.load(inputfile+'.npy')
    for idx, tasks in enumerate(tasksets):
        # if idx == 11:
        pp = PdfPages(folder + filePrefix + repr(idx) +'.pdf')
        results = []
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
            # elif mode == 5:
            #     # used to print out a continuous curve of results with different real value s
            #     ploting_with_s(n, por, fr, uti, filename, 1, 0, 100)
            else:
                raise NotImplementedError("Error: you use a mode without implementation")

if __name__=="__main__":
    main()
