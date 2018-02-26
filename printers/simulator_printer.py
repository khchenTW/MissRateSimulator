import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

N = 10
width = 0.15 # the width of the bars
ind = np.arange(N) # the x locations for the groups

#fileName = 'diffrent_set_size'
fileName = 'sims'
folder = 'final_plot/'
perfault = []

# plot in pdf
pp = PdfPages(folder + fileName + '.pdf')

percentageU = 75
#title = 'Tasks: '+ repr(2) + ', Utilization:'+repr(percentageU)+'%' + ', Fault Rate:'+repr(10**-4)
title = 'Tasks: '+ repr(2) + ', $U^N_{SUM}$:'+repr(percentageU)+'%' + ', Fault Rate:'+repr(10**-4)

plt.title(title, fontsize=20)
plt.grid(True)
plt.ylabel('Expected Miss Rate', fontsize=20)
ax = plt.subplot()
ax.set_yscale("log")
#ax.set_xlim([0, 11])
ax.set_ylim([10**-8,10**0])
ax.set_xticks(ind + width /2)
ax.set_xticklabels(('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'))
ax.tick_params(axis='both', which='major',labelsize=18)

# DAC18 submission
"""
SIM =[8.55e-05,8.75e-05,5.95e-05,0.000138,5.6e-05,8.75e-05,9.2e-05,8.3e-05,9.8e-05,0.0,]
CONV =  [0.000100020001,0.0001,0.000100009999001,0.000200009996,0.000100020006,0.00010000001,0.000100009999,0.00019999,0.0001,2.00089988018e-08,]
EMR = [0.0133025888398890,0.0120223624456771,0.00302096669656898,0.0296215691163006,0.0873980541349469,0.00109764161257549,0.0269409037578409,  0.107027821985454,0.00997987928221594,0.000116649831772272,]
"""
SIM= [8.6e-05,0.000207,0.000185,0.0001055,0.000109,0.000273,0.000181,0.000206,0.0,9.65e-05,]
CONV = [0.000100000001,0.000100040002999,0.0001,0.000100000001,0.000100000003999,0.000200009996,0.00019999,0.0001,2.99980009999e-08,0.000100020000999,]
EMR = [0.0422119642315498,0.0351124012287710,0.0592019701239771,0.0205348473070408,0.000156316786062342,0.0246925811940809,0.187875635377523,0.0517142950402505,0.000738483417064658,0.00876974639438917,]

try:
    pass
    rects1 = ax.bar(ind-0.1, SIM, width, color='black', edgecolor='black')
    rects2 = ax.bar(ind+0.1, CONV, width, fill=False, edgecolor='black')
    rects3 = ax.bar(ind+0.3, EMR, width, edgecolor='black', hatch="/")
    ax.legend((rects1[0], rects2[0], rects3[0]), ('SIM', 'CON', 'EMR'))
except ValueError:
    print "ValueError"
figure = plt.gcf()
figure.set_size_inches([10,6.5])

#plt.legend(handles=[set1, set2, set3, set4, set5], fontsize=12, frameon=True, loc=3)

pp.savefig()
plt.clf()
pp.close()

