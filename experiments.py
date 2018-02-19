from dispatcher import *
import sys
import numpy as np
import timing

faultRate = [10**-4]
#faultRate = [10**0]
power = [4]
h = 0
n = 10
sumbound = 3 #old setup for J'
utilization = [75]
#hardTaskFactor = [2.2/1.2]

def main():
    args = sys.argv
    if len(args) < 4:
        print "Usage: python experiments.py [mode] [count] [idx]"
        return -1
    mode = int(args[1])
    tasksets_amount = int(args[2])
    part = int(args[3])
    if mode == 0:
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                fileInput=taskSetInput(uti, fr, por, tasksets_amount, part)
                #print np.load(fileInput+'.npy')
    elif mode == 1:
        #use to quickly get emr
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                experiments_emr(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))
    elif mode == 2:
        #use to get sim results together with emr
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                experiments_sim(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))
    elif mode == 3:
        #use to get sim results together with emr
        for por, fr in enumerate(faultRate):
            for uti in utilization:
                trendsOfPhiMI(por, fr, uti,'inputs/'+str(uti)+'_'+str(power[por])+'_'+str(tasksets_amount)+'_'+str(part))

if __name__=="__main__":
    main()
