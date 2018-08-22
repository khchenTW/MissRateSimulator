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


hardTaskFactor = [1.83]

# Setting for Fig4:
faultRate = [10**-4]
power = [4]
utilization = [70]

# Setting for Fig5:
# faultRate = [10**-4]
# power = [4]
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

    jobnum = 2000000
    Outputs = True
    filePrefix = 'sim'
    folder = 'figures/'
    pp = PdfPages(folder + "task" + repr(n) + "-" + filePrefix + '.pdf')
    SimRateList = []
    ExpectedMissRate = []
    ConMissRate = []
    runtimeEMR = []
    runtimeCON = []
    runtimeSIM = []

    tasksets = np.load(inputfile+'.npy')
    tasksets_amount = len(tasksets)
    pass_amount = 0
    #try:
    filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
    SIM = np.load(filename+'.npy')
    filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
    runtimeSIM=np.load(filename+'.npy')
    filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
    EMR = np.load(filename+'.npy')
    filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
    runtimeEMR=np.load(filename+'.npy')
    filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
    CON = np.load(filename+'.npy')
    filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
    runtimeCON=np.load(filename+'.npy')
    #except IOError:
        #Outputs = False

    if Outputs is False:
        runtimeEMR = []
        runtimeCON = []
        runtimeSIM = []
        for idx, tasks in enumerate(tasksets):

            if pass_amount == 20:
                break
            global lookupTable
            global conlookupTable

            print "Running simulator"
            simulator=MissRateSimulator(n, tasks)

            # EPST + Theorem2
            # report the miss rate of the lowest priority task

            t1 = time.clock()
            simulator.dispatcher(jobnum, fr)
            tmp = simulator.missRate(n-1)
            diff = time.clock()-t1
            print ("--- Simulation %s seconds ---" % diff)
            runtimeSIM.append(diff)


            if tmp < 10**-5:
                continue
            else:
                pass_amount += 1
                SimRateList.append(tmp)
                print "Approximate the miss rate"

                t1 = time.clock()
                ExpectedMissRate.append(NewApproximation(n, fr, sumbound, n-1, tasks, 0))
                diff = time.clock()-t1
                print ("--- EMR %s seconds ---" % diff)
                runtimeEMR.append(diff)
                t1 = time.clock()
                ConMissRate.append(NewApproximation(n, fr, sumbound, n-1, tasks, 1))
                diff = time.clock()-t1
                print ("--- CON %s seconds ---" % diff)
                runtimeCON.append(diff)


        filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
        np.save(filename, SimRateList)
        SIM = np.load(filename+'.npy')
        filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_Sim'
        np.save(filename, runtimeSIM)

        filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, ExpectedMissRate)
        EMR = np.load(filename+'.npy')
        filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, runtimeEMR)

        filename = 'outputs/sim'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, ConMissRate)
        CON = np.load(filename+'.npy')
        filename = 'outputs/simt'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, runtimeCON)

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SIM
    print "ExpectedMissRate:"
    print EMR
    print "ConMissRate:"
    print CON

    print ("EMR Avg %s seconds" %(reduce(lambda y, z: y + z, runtimeEMR)/len(runtimeEMR)))
    print runtimeEMR
    print ("CON Avg %s seconds" %(reduce(lambda y, z: y + z, runtimeCON)/len(runtimeCON)))
    print runtimeCON
    print ("SIM Avg %s seconds" %(reduce(lambda y, z: y + z, runtimeSIM)/len(runtimeSIM)))
    print runtimeSIM


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
    title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$: '+repr(uti)+'\%' + ', $P_i^A$: '+repr(fr)
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
        rects1 = plt.bar(ind-0.1, SIM, width, edgecolor='black')
        rects2 = plt.bar(ind+0.1, CON, width, edgecolor='black')
        rects3 = plt.bar(ind+0.3, EMR, width, edgecolor='black')
        plt.legend((rects1[0], rects2[0], rects3[0]),('SIM', 'CON', 'AB'), ncol=3, loc=9, bbox_to_anchor=(0.5, 1), prop={'size':20})
    except ValueError:
        print "Value ERROR!!!!!!!!!!"
    figure = plt.gcf()
    figure.set_size_inches([14.5,6])

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
    folder = 'figures/'
    # folder = '/home/khchen/Dropbox/'
    upperJ = 7

    tasksets = np.load(inputfile+'.npy')
    tasksets_amount = len(tasksets)
    EMRResults = []
    CONResults = []

    runtimeEMR = []
    runtimeCON = []
    try:
        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        runtimeEMR = np.load(filename+'.npy')
        filename = 'outputs/trendsR'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        EMRResults = np.load(filename+'.npy')

        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        runtimeCON = np.load(filename+'.npy')
        filename = 'outputs/trendsR'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        CONResults = np.load(filename+'.npy')
    except IOError:
        Outputs = False


    if Outputs is False:
        runtimeEMR = []
        runtimeCON = []
        EMRResults = []
        CONResults = []
        for x in range(1, upperJ):
            stampPHIEMR = []
            stampPHICON = []
            probPHIEMR = []
            probPHICON = []
            print x
            for tasks in tasksets:
                t1 = time.clock()
                prepareTable(n, fr, x, n-1, tasks, 0)
                r = wRoutine(x,0)
                t2 = time.clock()
                diff = t2-t1
                print ("--- Chernoff %s seconds ---" % diff)
                stampPHIEMR.append(diff)
                probPHIEMR.append(r)

                if n >= 10 and x > upperJ-2:
                    continue
                else:
                    probs = []
                    states = []
                    pruned = []
                    t3 = time.clock()
                    prepareTable(n, fr, x, n-1, tasks, 1)
                    c = wRoutine(x, 1)
                    t4 = time.clock()
                    diff = t4-t3
                    print ("--- CON %s seconds ---" % diff)
                    stampPHICON.append(diff)
                    probPHICON.append(c)
            runtimeEMR.append(stampPHIEMR)
            EMRResults.append(probPHIEMR)
            if len(stampPHICON) > 0:
                runtimeCON.append(stampPHICON)
                CONResults.append(probPHICON)

        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, runtimeEMR)
        filename = 'outputs/trendsR'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_EMR'
        np.save(filename, EMRResults)
        filename = 'outputs/trends'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, runtimeCON)
        filename = 'outputs/trendsR'+str(n)+'_'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_CON'
        np.save(filename, CONResults)

    print "for runtime:"
    for x, timeS in enumerate(runtimeEMR):
        print ("EMR Avg %s seconds for index %s" %(reduce(lambda y, z: y + z, timeS)/len(timeS), x+1))
        print ("x:%s" % (x+1)) , timeS
    for x, timeS in enumerate(runtimeCON):
        print ("CON Avg %s seconds for index %s" %(reduce(lambda y, z: y + z, timeS)/len(timeS), x+1))
        print ("x:%s" % (x+1)) , timeS

    print "for results:"
    for x, resS in enumerate(EMRResults):
        print ("EMR Avg %s prob for index %s" %(reduce(lambda y, z: y + z, resS)/len(resS), x+1))
        print ("x:%s" % (x+1)) , resS
    for x, resS in enumerate(CONResults):
        print ("CON Avg %s prob for index %s" %(reduce(lambda y, z: y + z, resS)/len(resS), x+1))
        print ("x:%s" % (x+1)) , resS

    filePrefix = 'trends'
    pp = PdfPages(folder + "task" + repr(n) + "-" + filePrefix + '.pdf')
    # Label
    title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$: '+repr(uti)+'\%' + ', $P_i^A$: '+repr(fr)

    ##the blue box
    #boxprops = dict(linewidth=2, color='blue')
    ##the median line
    #medianprops = dict(linewidth=2.5, color='red')
    #whiskerprops = dict(linewidth=2.5, color='black')
    #capprops = dict(linewidth=2.5)

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

    rects1=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR], '--o' )
    if n == 10:
        labels = [j for j in range(1, upperJ-1)]
        print labels
    rects2=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON], '-.D' )
    plt.legend((rects1[0], rects2[0]),('AB','CON'), prop={'size': 20})

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

    #plt.show()
    pp.savefig()
    plt.clf()
    pp.close()

    """
    For trends-prob
    """

    filePrefix = 'trends-prob'
    pp = PdfPages(folder + "task" + repr(n) + "-" + filePrefix + '.pdf')
    title = 'Tasks: '+ repr(n) + ', $U^N_{SUM}$: '+repr(uti)+'\%' + ', $P_i^A$: '+repr(fr)

    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.ylabel('Probability $\Phi_{k, j}$', fontsize=20)
    plt.xlabel('Step j', fontsize=22)
    plt.yscale("log")
    # ax.set_ylim([10**-28,10**0])
    print len(EMRResults)
    print len(CONResults)

    labels = [j for j in range(1, upperJ)]
    print labels
    plt.xticks(labels)

    rects1=plt.plot(labels, [float(reduce(lambda y, z: y + z, resS)/len(resS)) for resS in EMRResults], '--o' )
    if n == 10:
        labels = [j for j in range(1, upperJ-1)]
        print labels
    rects2=plt.plot(labels, [float(reduce(lambda y, z: y + z, resS)/len(resS)) for resS in CONResults], '-.D' )
    plt.legend((rects1[0], rects2[0]),('AB','CON'), prop={'size': 20})

    # Figure scale

    figure = plt.gcf()
    figure.set_size_inches([10,6.5])


    pp.savefig()
    plt.clf()
    pp.close()


def experiments_art(n, por, fr, uti, inputfile):
    # this is for artifical test (motivational example)
    jobnum = 5000000
    SimRateList = []
    tasksets = []
    sets = 100

    n = 2
    tasks = []
    tasks.append({'period': 3, 'abnormal_exe': 2, 'deadline': 3, 'execution': 2, 'prob': 0})
    tasks.append({'period': 5, 'abnormal_exe': 2.25, 'deadline': 5, 'execution': 1, 'prob': 5e-01})

    for x in range(sets):
        tasksets.append(tasks)

    for ind, tasks in enumerate(tasksets):

        simulator=MissRateSimulator(n, tasks)
        simulator.dispatcher(jobnum, 0.5)
        SimRateList.append(simulator.missRate(n-1))
        print ind, simulator.missRate(n-1)

    print "Result for fr"+str(power[por])+"_uti"+str(uti)
    print "SimRateList:"
    print SimRateList
    print "AVG: ", reduce(lambda x, y: x+y, SimRateList) / len(SimRateList)
    print "MAX: ", max(SimRateList)
    maxResult = max(SimRateList)
    AVGResult = max(SimRateList)


    ofile = "txt/ART_task"+str(n)+".txt"
    fo = open(ofile, "wb")
    fo.write("Simulation Miss Rate:")
    fo.write("\n")
    fo.write(str(SimRateList))
    fo.write(AVGResult)
    fo.write(maxResult)
    fo.write("\n")
    fo.close()


def ploting_together():
    filePrefix = 'plots-'
    folder = 'figures/'
    runtimeEMR5 = []
    runtimeCON5 = []
    EMRResults5 = []
    CONResults5 = []
    runtimeEMR10 = []
    runtimeCON10 = []
    EMRResults10 = []
    CONResults10 = []
    diffResults10 = []
    diffResults5 = []

    try:
        filename = 'outputs/backup/trends'+str(5)+'_'+str(70)+'_'+str(4)+'_'+str(10)+'_EMR'
        runtimeEMR5 = np.load(filename+'.npy')
        filename = 'outputs/backup/trendsR'+str(5)+'_'+str(70)+'_'+str(4)+'_'+str(10)+'_EMR'
        EMRResults5 = np.load(filename+'.npy')

        filename = 'outputs/backup/trends'+str(10)+'_'+str(70)+'_'+str(4)+'_'+str(5)+'_EMR'
        runtimeEMR10 = np.load(filename+'.npy')
        filename = 'outputs/backup/trendsR'+str(10)+'_'+str(70)+'_'+str(4)+'_'+str(5)+'_EMR'
        EMRResults10 = np.load(filename+'.npy')

        filename = 'outputs/backup/trends'+str(5)+'_'+str(70)+'_'+str(4)+'_'+str(10)+'_CON'
        runtimeCON5 = np.load(filename+'.npy')
        filename = 'outputs/backup/trendsR'+str(5)+'_'+str(70)+'_'+str(4)+'_'+str(10)+'_CON'
        CONResults5 = np.load(filename+'.npy')

        filename = 'outputs/backup/trends'+str(10)+'_'+str(70)+'_'+str(4)+'_'+str(5)+'_CON'
        runtimeCON10 = np.load(filename+'.npy')
        filename = 'outputs/backup/trendsR'+str(10)+'_'+str(70)+'_'+str(4)+'_'+str(5)+'_CON'
        CONResults10 = np.load(filename+'.npy')

        for EMRx, CONy in zip(EMRResults5, CONResults5):
            diffResults5.append(
                (reduce(lambda y, z: y + z, EMRx)/len(EMRx))/(reduce(lambda y, z: y + z, CONy)/len(CONy))
            )


        for EMRx, CONy in zip(EMRResults10, CONResults10):
            diffResults10.append(
               (reduce(lambda y, z: y + z, EMRx)/len(EMRx))/(reduce(lambda y, z: y + z, CONy)/len(CONy))
            )



        filePrefix = 'trends-together'
        pp = PdfPages(folder + filePrefix + '.pdf')
        # Label
        title = '$U^N_{\small\mbox{SUM}}$: '+repr(70)+'\%' + ', $P_i^A$: '+repr(10**-4)

        plt.title(title, fontsize=20)
        plt.grid(True)
        plt.ylabel('Average Analysis Runtime (seconds)', fontsize=20)
        plt.xlabel('Step j for $\Phi_{k, j}$', fontsize=22)
        # plt.yscale("log")
        plt.ylim([-300,6000])
        plt.xlim([0.5,6.5])

        labels = [j for j in range(1, 7)]
        plt.xticks(labels)


        rects1=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR5], '--o', ms=7 )
        rects2=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON5], '-.D', ms=7 )
        rects3=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR10], '--p', ms=7 )
        labels = [j for j in range(1, 6)]
        rects4=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON10], '-.v', ms=7 )
        plt.legend((rects1[0], rects2[0], rects3[0], rects4[0]),('AB-task5','CON-task5','AB-task10','CON-task10'), prop={'size': 20}, loc=2)

        # Figure scale

        figure = plt.gcf()
        figure.set_size_inches([10,5.5])

        #plt.show()
        pp.savefig()
        plt.clf()
        pp.close()

        # blank figure
        pp = PdfPages(folder + "trends-blank" + '.pdf')
        title = '$U^N_{\small\mbox{SUM}}$: '+repr(70)+'\%' + ', $P_i^A$: '+repr(10**-4)

        plt.title(title, fontsize=20)
        plt.grid(True)
        plt.ylabel('Average Analysis Runtime (seconds)', fontsize=20)
        plt.xlabel('Step j for $\Phi_{k, j}$', fontsize=22)
        plt.ylim([-300,6000])
        plt.xlim([0.5,6.5])
        # plt.yscale("log")
        # ax.set_ylim([10**-28,10**0])

        labels = [j for j in range(1, 7)]
        plt.xticks(labels)
        figure = plt.gcf()
        figure.set_size_inches([10,5.5])

        pp.savefig()
        plt.clf()
        pp.close()


        filePrefix = 'trends-together-only5'
        pp = PdfPages(folder + filePrefix + '.pdf')
        # Label
        title = '$U^N_{\small\mbox{SUM}}$: '+repr(70)+'\%' + ', $P_i^A$: '+repr(10**-4)

        plt.title(title, fontsize=20)
        plt.grid(True)
        plt.ylabel('Average Analysis Runtime (seconds)', fontsize=20)
        plt.xlabel('Step j for $\Phi_{k, j}$', fontsize=22)
        plt.ylim([-300,6000])
        plt.xlim([0.5,6.5])
        # plt.yscale("log")
        # ax.set_ylim([10**-28,10**0])

        labels = [j for j in range(1, 7)]
        plt.xticks(labels)

        rects1=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR5], '--o', ms=7)
        rects2=plt.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON5], '-.D', ms=7)
        plt.legend((rects1[0], rects2[0]),('AB-task5','CON-task5'), prop={'size': 20}, loc=2)

        # Figure scale

        figure = plt.gcf()
        figure.set_size_inches([10,5.5])

        #plt.show()
        pp.savefig()
        plt.clf()
        pp.close()

        '''TrendsR together'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,hspace =0.15)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


        filePrefix = 'trendsR-together'
        pp = PdfPages(folder + filePrefix + '.pdf')
        title = '$U^N_{\small\mbox{SUM}}$: '+repr(70)+'\%' + ', $P_i^A$: '+repr(10**-4)
        plt.title(title, fontsize=20)
        for i in range(1, 3):
            ax = fig.add_subplot(2,1,i)
            if i == 1:
                ax.set_ylabel('Avg Analysis Runtime (sec)', fontsize=15)
                # ax.set_xlabel('Step j for $\Phi_{k, j}$', fontsize=15)
                ax.set_ylim([-300,6000])
                ax.set_xlim([0.5,6.5])

                labels = [j for j in range(1, 7)]
                # ax.set_xticks(labels)

                rects1=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR5], '--o' , ms=7)
                rects2=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON5], '-.D' , ms=7)
                rects3=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR10], '--p', ms=7)
                labels = [j for j in range(1, 6)]
                rects4=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON10], '-.v', ms=7)
                ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),('AB-task5','CON-task5','AB-task10','CON-task10'), prop={'size': 15}, loc=2)
                ax.grid()
            else:

                # Label
                ax.set_ylabel('$\Delta = \Phi_{k,j}^{\mbox{AB}} / \Phi_{k,j}^{\mbox{CON}}$', fontsize=15)
                ax.set_xlabel('Step j for $\Phi_{k, j}$', fontsize=20)
                ax.set_yscale("log")
                ax.set_ylim([10**0, 10**10])
                ax.set_xlim([0.5,6.5])

                labels = [j for j in range(1, 7)]
                ax.set_xticks(labels)


                rects1=ax.plot(labels, diffResults5, '--o', ms=7)
                labels = [j for j in range(1, 6)]
                rects2=ax.plot(labels, diffResults10, '-.D', ms=7 )
                ax.legend((rects1[0], rects2[0]),('Diff-task5','Diff-task10'), prop={'size': 15}, loc=2)

                ax.grid()

        #plt.show()
        figure = plt.gcf()
        figure.set_size_inches([10,5.5])
        pp.savefig()
        plt.clf()
        pp.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.9,left=0.1,right=0.95,hspace =0.15)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


        filePrefix = 'trendsR-blank'
        pp = PdfPages(folder + filePrefix + '.pdf')
        title = '$U^N_{\small\mbox{SUM}}$: '+repr(70)+'\%' + ', $P_i^A$: '+repr(10**-4)
        plt.title(title, fontsize=20)
        for i in range(1, 3):
            ax = fig.add_subplot(2,1,i)
            if i == 1:
                ax.set_ylabel('Avg Analysis Runtime (sec)', fontsize=15)
                # ax.set_xlabel('Step j for $\Phi_{k, j}$', fontsize=15)
                ax.set_ylim([-300,6000])
                ax.set_xlim([0.5,6.5])

                labels = [j for j in range(1, 7)]
                # ax.set_xticks(labels)


                rects1=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR5], '--o' , ms=7)
                rects2=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON5], '-.D' , ms=7)
                rects3=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeEMR10], '--p', ms=7)
                labels = [j for j in range(1, 6)]
                rects4=ax.plot(labels, [float(reduce(lambda y, z: y + z, timeS)/len(timeS)) for timeS in runtimeCON10], '-.v', ms=7)
                ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),('AB-task5','CON-task5','AB-task10','CON-task10'), prop={'size': 15}, loc=2)

                # Figure scale

                # figure = ax.gcf()
                # figure.set_size_inches([10,5.5])

                ax.grid()
            else:

                # Label
                ax.set_ylabel('$\Delta = \Phi_{k,j}^{\mbox{AB}} / \Phi_{k,j}^{\mbox{CON}}$', fontsize=15)
                ax.set_xlabel('Step j for $\Phi_{k, j}$', fontsize=20)
                ax.set_yscale("log")
                ax.set_ylim([10**0, 10**10])
                ax.set_xlim([0.5,6.5])

                labels = [j for j in range(1, 7)]
                ax.set_xticks(labels)


                # rects1=ax.plot(labels, diffResults5, '--o' )
                # labels = [j for j in range(1, 5)]
                # rects2=ax.plot(labels, diffResults10, '-.D' )
                # ax.legend((rects1[0], rects2[0]),('Diff-task5','Diff-task10'), prop={'size': 15}, loc=2)

                ax.grid()

        #plt.show()
        figure = plt.gcf()
        figure.set_size_inches([10,5.5])
        pp.savefig()
        plt.clf()
        pp.close()

    except IOError:
        print "Inputs have error"

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
                # used to print out the figure presented in the slides
                ploting_together()
            else:
                raise NotImplementedError("Error: you use a mode without implementation")

if __name__=="__main__":
    main()
