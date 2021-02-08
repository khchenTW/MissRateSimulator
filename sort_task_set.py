# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

from __future__ import division
import random
import math
import numpy
import sys, getopt
import operator

def sort(tasks, criteria):
    return sorted(tasks, key=lambda item:item[criteria])

def sortEvent(tasks, criteria):
    return sorted(tasks, key=operator.attrgetter(criteria))
