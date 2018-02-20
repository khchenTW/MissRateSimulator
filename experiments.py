from dispatcher import *
import sys
import numpy as np
import timing

faultRate = [10**-4]
# this list is used to generate a readible name of output.
power = [4]
utilization = [75]

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
