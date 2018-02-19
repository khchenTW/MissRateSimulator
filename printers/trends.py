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




#fileName = 'diffrent_set_size'
fileName = 'trends'
folder = 'final_plot/'
perfault = []

# plot in pdf
pp = PdfPages(folder + fileName + '.pdf')

percentageU = 75
title = 'Tasks: '+ repr(10) + ', $U^N_{SUM}$:'+repr(percentageU)+'%' + ', Fault Rate:'+repr(10**-4)

plt.title(title, fontsize=20)
plt.grid(True)
plt.ylabel('$\Phi_{k,j}$', fontsize=20)
plt.xlabel('j', fontsize=22)
ax = plt.subplot()
ax.set_yscale("log")
ax.set_xlim([0, 11])
ax.set_ylim([10**-105,10**10])
ax.tick_params(axis='both', which='major',labelsize=18)


x1 = [x+1 for x in range(10)]
y1 = [4.85321271133951e-8, 1.76446939583656e-19, 7.27765501246738e-33, 3.55147656039182e-42, 1.32905669103551e-46, 8.07315175282262e-66, 1.94437868433586e-73, 2.81856379808910e-83, 1.29697126441990e-87, 2.24248374704863e-91]

y2 = [0.0352508010809133, 6.75655947962359e-17, 5.47107719608742e-24, 7.18410327502350e-35, 2.72714740069219e-35, 3.76367161139018e-45, 2.79954615654100e-53, 2.71409087167204e-65, 7.66494686965442e-72, 4.47955070314941e-75]
y3 = [0.000479771691708255, 3.20437610677642e-21, 4.01513158708227e-32, 1.12283078592791e-38, 3.77695649097479e-50, 1.06131114613541e-55, 6.33965733852326e-67, 1.27885540597455e-72, 1.31038431090129e-83, 3.18625979107394e-89]

y4 = [0.0439059454945871, 6.49319024126218e-10, 1.49167894794263e-15, 1.95212079714248e-21, 2.41574732217838e-29, 1.78583528961877e-37, 2.48234095193412e-42, 7.35715422189012e-48, 1.07960691062464e-57, 3.91109813706627e-58]

y5 = [5.83229986785099e-6, 7.41309980003440e-26, 1.04482910154417e-38, 1.43201875514101e-48, 7.31464892943305e-57, 3.49682120425866e-65, 3.60145304933650e-80, 3.34097656839234e-90, 5.58195386186468e-96, 5.56166773970043e-100]
try:
    set1,=ax.plot(x1, y1, 'r*', label='Set1')
    set2,=ax.plot(x1, y2, 'bo', label='Set2')
    set3,=ax.plot(x1, y3, 'g^', label='Set3')
    set4,=ax.plot(x1, y4, 'bs', label='Set4')
    set5,=ax.plot(x1, y5, 'rx', label='Set5')
except ValueError:
    print "ValueError"
figure = plt.gcf()
figure.set_size_inches([10,6.5])

plt.legend(handles=[set1, set2, set3, set4, set5], fontsize=12, frameon=True, loc=3)

pp.savefig()
plt.clf()
pp.close()

