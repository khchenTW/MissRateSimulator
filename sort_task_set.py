from __future__ import division
import random
import math
import numpy
import sys, getopt

def sort(tasks, criteria):
    return sorted(tasks, key=lambda item:item[criteria])

def sortEvent(tasks, criteria):
    return sorted(tasks, key=lambda item:item.criteria)
