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


hardTaskFactor = [1.83]

# Setting for Fig4:
faultRate = [10**-4]
power = [4]
utilization = [70]

# Setting for Fig5:
#faultRate = [10**-4]
#power = [4]
#utilization = [75]

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
#jobnum = 2000000
jobnum = 2
lookupTable = []
conlookupTable = []

# for task generation
def taskGeneWithTDA(uti, fr, n):
    while (1):
        #tasks = task_generator.taskGeneration_p(n,uti)
        tasks = task_generator.taskGeneration_rounded(n, uti)
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

#def lookup(k, tasks, fr, numDeadline, mode):
#    global lookupTable
#    global conlookupTable
#    if mode == 0:
#        if lookupTable[k][numDeadline] == -1:
#            lookupTable[k][numDeadline] = EPST.probabilisticTest_s(k, tasks, numDeadline, SympyChernoff, -1)
#        return lookupTable[k][numDeadline]
#    else:
#        if conlookupTable[k][numDeadline] == -1:
#            #for ECRTS
#            probs = []
#            states = []
#            pruned = []
#            conlookupTable[k][numDeadline] = deadline_miss_probability.calculate_pruneCON(tasks, fr, probs, states, pruned, numDeadline)
#        return conlookupTable[k][numDeadline]

#def Approximation(n, fr, J, k, tasks, mode=0):
#    # mode 0 == EMR, 1 = CON
#    # J is the bound of the idx
#    if mode == 0:
#        global lookupTable
#        lookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]
#    else:
#        global conlookupTable
#        conlookupTable = [[-1 for x in range(sumbound+2)] for y in range(n)]

#    probsum = 0
#    # this is the summation from 1 to J (prob * j)
#    for x in range(1, J+1):
#        probsum += lookup(k, tasks, fr, x, mode)*x

#    if probsum == 0:
#        return 0
#    else:
#        #print 1/(1+(1-lookup(k, tasks, 1))/probsum)
#        #print probsum/(1+probsum-lookup(k, tasks, 1))
#        #for avoiding numerical inconsistance
#        return probsum/(1+probsum-lookup(k, tasks, fr, 1, mode))

def NewApproximation(n, fr, J, k, tasks, mode=0):
    # mode 0 == EMR, 1 = CON
    # J is the bound of the idx
    prepareTable(n, fr, J, n-1, tasks, mode)
    probsum = 0.
    phikl = np.float128(1.0)
    # this is the summation from 1 to J (prob * j)
    #print "which mode:", mode
    for x in range(1, J+1):
        #print "for index:", x
        phikl = wRoutine(x, mode)
        # print phikl
        probsum += phikl*x

    if probsum == 0:
        return 0
    else:
        #for avoiding numerical inconsistance
        return probsum/(1+probsum-wRoutine(1, mode))

def wRoutine(l, mode=0):
    #suppose that the table is ready
    #this is the Lemma 1 in the paper
    resProb = np.float128(0.)
    listProb = []
    if l == 0:
        return 1.0
    else:
        for w in range(1, l+1):
            if mode == 0:
                resProb = lookupTable[w]*wRoutine(l-w, mode)
            else:
                resProb = conlookupTable[w]*wRoutine(l-w, mode)
            # if resProb == 0.0:
            #     print "resProb is 0", w, lookupTable[w], wRoutine(l-w, mode)
            listProb.append(resProb)
    # print "l is ", l,  listProb
    return max(listProb)

def prepareTable(n, fr, J, k, tasks, mode=0):
    tmpList = []
    if mode == 0:
        global lookupTable
        lookupTable = [-1 for x in range(J+1)]
        hpTasks = tasks[:k]
        for x in range(1, J+1):
            #here calculate the \phi^\theta_{k, x}
            tmpList = EPST.ktda_k(tasks[k], hpTasks, x, SympyChernoff, -1)
            #tmpList = EPST.ktda_k(tasks[k], hpTasks, x, Chernoff_bounds, 1)
            print EPST.ktda_k(tasks[k], hpTasks, x, SympyChernoff, -1)
            #print EPST.ktda_k(tasks[k], hpTasks, x, Chernoff_bounds, 1)
            lookupTable[x] = tmpList[0]
    else:
        global conlookupTable
        conlookupTable = [-1 for x in range(J+1)]
        for x in range(1, J+1):
            #here calculate the \phi^\theta_{k, x}
            #for ECRTS
            probs = []
            states = []
            pruned = []
            conlookupTable[x] = deadline_miss_probability.calculate_pruneCON(tasks, fr, probs, states, pruned, x)

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


            print "Approximate the miss rate"
            tmp = NewApproximation(n, fr, sumbound, n-1, tasks, 0)
            if tmp < 10**-4:
                continue
            else:
                pass_amount += 1
                ExpectedMissRate.append(tmp)
                ConMissRate.append(NewApproximation(n, fr, sumbound, n-1, tasks, 1))

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
    ind = np.arange(len(EMR))
    #prune for leq 20 sets
    print "Num of SIM:",len(SIM)
    print "Num of CON:",len(CON)
    print "Num of EMR:",len(EMR)
    if len(EMR) > 20:
        SIM = SIM[:20]
        EMR = EMR[:20]
        CON = CON[:20]
        ind = np.arange(20) # the x locations for the groups

    width = 0.15
    title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$:'+repr(uti)+'%' + ', Fault Rate:'+repr(fr)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.ylabel('Expected Miss Rate', fontsize=20)
    plt.yscale("log")
    plt.ylim([10**-5, 10**0])
    pltlabels = []
    for idt, tt in enumerate(EMR):
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
        ExpectedMissRate.append(NewApproximation(n, fr, sumbound, n-1, tasks, 0))

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

    Outputs = True
    filePrefix = 'trends'
    folder = 'figures/'
    pp = PdfPages(folder + "task" + repr(n) + "-" + filePrefix + '.pdf')
    #upperJ = 11
    upperJ = 4

    tasksets = np.load(inputfile+'.npy')
    tasksets_amount = len(tasksets)
    CResults = []
    Results = []
    xResults = []

    runtimeEMR = []
    runtimeCON = []
    try:
        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        runtimeEMR = np.load(filename+'.npy')
        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        runtimeCON = np.load(filename+'.npy')
    except IOError:
        Outputs = False


    if Outputs is False:
        runtimeEMR = []
        runtimeCON = []
        for x in range(1, upperJ):
        # for x in range(1, 4):
            stampPHIEMR = []
            stampPHICON = []
            print x
            for tasks in tasksets:
                t1 = time.clock()
                prepareTable(n, fr, x, n-1, tasks, 0)
                r = wRoutine(x,0)
                t2 = time.clock()
                diff = t2-t1
                print ("--- Chernoff %s seconds ---" % diff)
                stampPHIEMR.append(diff)

                Results.append(r)
                xResults.append(r*x)
                #if x < upperJ-3:
                if x < 3:
                    probs = []
                    states = []
                    pruned = []
                    t3 = time.clock()
                    prepareTable(n, fr, x, n-1, tasks, 1)
                    # c = deadline_miss_probability.calculate_pruneCON(tasks, fr, probs, states, pruned, x)
                    c = wRoutine(x, 1)
                    t4 = time.clock()
                    diff = t4-t3
                    print ("--- CON %s seconds ---" % diff)
                    stampPHICON.append(diff)
                    CResults.append(c)
            runtimeEMR.append(stampPHIEMR)
            # reduce(lambda y, z: y + z, stampPHIEMR)/len(stampPHIEMR)
            if len(stampPHICON) > 0:
                runtimeCON.append(stampPHICON)
                # reduce(lambda y, z: y + z, stampPHICON)/len(stampPHICON)

        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, runtimeEMR)
        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, runtimeCON)

    for x, timeS in enumerate(runtimeEMR):
        print ("EMR Avg %s seconds for index %s" %(reduce(lambda y, z: y + z, timeS)/len(timeS), x+1))
        print ("x:%s" % (x+1)) , timeS
    for x, timeS in enumerate(runtimeCON):
        print ("CON Avg %s seconds for index %s" %(reduce(lambda y, z: y + z, timeS)/len(timeS), x+1))
        print ("x:%s" % (x+1)) , timeS


    # # collect results
    # xlistRes.append(xResults)
    # ClistRes.append(CResults)
    # listRes.append(Results)

    # Label
    title = 'Tasks:' + repr(n) + ', $U^N_{SUM}$:'+repr(uti)+'%'

    #the blue box
    boxprops = dict(linewidth=2, color='blue')
    #the median line
    medianprops = dict(linewidth=2.5, color='red')
    whiskerprops = dict(linewidth=2.5, color='black')
    capprops = dict(linewidth=2.5)

    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.ylabel('Analysis Runtime (seconds)', fontsize=20)
    plt.xlabel('Step j', fontsize=22)
    # plt.yscale("log")
    # ax.set_ylim([10**-28,10**0])
    print len(runtimeEMR)
    print len(runtimeCON)

    labels = [j for j in range(1, upperJ)]
    print labels
    plt.xticks(labels)

    #plt.violinplot([x for x in runtimeEMR], showmedians=True, showmeans=False)
    #plt.violinplot([x for x in runtimeCON], showmedians=True, showmeans=False)

    #plt.errorbar(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR], yerr=1, fmt='--o' )


    # labels = [j for j in range(1, 3)]
    # print labels
    # plt.boxplot([x for x in runtimeCON], 0 , '', labels=labels, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    #plt.boxplot(runtimeEMR, 0 , '', labels=labels, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    # plt.boxplot(runtimeCON, 0 , '', labels=labels, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    #except ValueError:



    # Figure scale

    figure = plt.gcf()
    figure.set_size_inches([10,6.5])

    # Legend

    # box = mpatches.Patch(color='blue', label='First to Third Quartiles', linewidth=3)
    # av = mpatches.Patch(color='red', label='Median', linewidth=3)
    # whisk = mpatches.Patch(color='black', label='Whiskers', linewidth=3)

    # plt.legend(handles=[av, box, whisk], fontsize=16, frameon=True, loc=1)

    plt.show()
    pp.savefig()
    plt.clf()
    pp.close()


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
