''' Implementation of the novel multinomial-based approach as detailed in Section 5.
For the actual implementation the binomial case is considered.
Furthermore, since the multinomial-based approach relies on convolution to merge
multinomial representations of tasks the convolution based approach is implemented
in this file as well. '''

from __future__ import division
import random
import math
import numpy as np
from operator import itemgetter, attrgetter
from pkg_resources import get_distribution

import TDA


''' Calculates the probability of deadline miss as detailed in Section 5.
All job releases of higher priority tasks are considered.

'tasks represents' the given task set,
'prob_abnormal' the probability of abnormal execution, i.e., higher WCET.
'probabilities' tracks the calculated probabilities for each time point
'states' tracks the number of states considered for each time point '''
def calculate(tasks, prob_abnormal, probabilties, states):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    times = []
    for i in all_times:
        if i > min_time:
            times.append(i)
    times.sort()
    for time in times:
        prob = calculate_probabiltiy(tasks, time, prob_abnormal, states)
        probabilties.append(prob)
    probability = 1
    for i in range(0, len(times),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' KHCHEN: extension for consecutive deadlines, which calculates the probability of deadline miss as detailed in Section 5.
All job releases of higher priority tasks are considered.
The state pruning introduced in Section 6.1 is used

'tasks represents' the given task set,
'prob_abnormal' the probability of abnormal execution, i.e., higher WCET.
'probabilities' tracks the calculated probabilities for each time point
'states' tracks the number of states for each time point
'pruned' tracks the number of states for each time point after pruning '''
def calculate_pruneCON(tasks, prob_abnormal, probabilties, states, pruned, numD):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']*numD
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    times = []
    for i in all_times:
        if i > min_time:
            times.append(i)
    times.sort()
    for time in times:
        prob = calculate_probabiltiy_prune(tasks, time, prob_abnormal, states, pruned)
        probabilties.append(prob)
    probability = 1
    for i in range(0, len(times),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' Calculates the probability of deadline miss as detailed in Section 5.
All job releases of higher priority tasks are considered.
The state pruning introduced in Section 6.1 is used

'tasks represents' the given task set,
'prob_abnormal' the probability of abnormal execution, i.e., higher WCET.
'probabilities' tracks the calculated probabilities for each time point
'states' tracks the number of states for each time point
'pruned' tracks the number of states for each time point after pruning '''
def calculate_prune(tasks, prob_abnormal, probabilties, states, pruned):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    times = []
    for i in all_times:
        if i > min_time:
            times.append(i)
    times.sort()
    for time in times:
        prob = calculate_probabiltiy_prune(tasks, time, prob_abnormal, states, pruned)
        probabilties.append(prob)
    probability = 1
    for i in range(0, len(times),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' Calculates the probability of deadline miss as detailed in Section 5.
All job releases of higher priority tasks are considered.
The state pruning introduced in Section 6.1 is used
Furthermore, the reduction of the binomial representation in Section 6.2

'max_error_allowed' describes the maximum total error that is allowed after state
reduction for all binomial representations.
'tasks represents' the given task set,
'prob_abnormal' the probability of abnormal execution, i.e., higher WCET.
'probabilities' tracks the calculated probabilities for each time point
'states' tracks the number of states for each time point
'pruned' tracks the number of states for each time point after pruning
'max_error' tracks the maximum possible error that can occur after the reduction '''
def calculate_prune_reduct(tasks, prob_abnormal, probabilties, states, pruned, max_error, max_error_allowed):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    times = []
    for i in all_times:
        if i > min_time:
            times.append(i)
    times.sort()
    error = []
    for time in times:
        prob = calculate_probabiltiy_prune_reduct(tasks, time, prob_abnormal, states, pruned, error, max_error_allowed)
        probabilties.append(prob)
    max_error.append(np.max(error))
    probability = 1
    for i in range(0, len(times),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' Approximates the probability of deadline miss as detailed in Section 5
by only considering the last release of all higher priority tasks.
The state pruning introduced in Section 6.1 is used

'tasks represents' the given task set,
'prob_abnormal' the probability of abnormal execution, i.e., higher WCET.
'probabilities' tracks the calculated probabilities for each time point
'states' tracks the number of states for each time point
'pruned' tracks the number of states for each time point after pruning '''
def approximate_prune(tasks, prob_abnormal, probabilties, states, pruned):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    tasks = sort(tasks, 'execution', True)
    times = last_release(tasks, deadline)
    times.sort()
    for time in times:
        prob = calculate_probabiltiy_prune(tasks, time, prob_abnormal, states, pruned)
        probabilties.append(prob)
    probability = 1
    for i in range(0, len(times),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' KHCHEN: Extension for consecutive deadline misses based on the convolution based approach by Maxim and Cucu-Grosjean [17]'''
def convolutionCON(tasks, prob_abnormal, probabilties, states, numD):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']*numD
    releases = []
    times = []
    calculate_releases(tasks, deadline, releases, prob_abnormal)
    releases = sorted(releases, key=lambda release:release[0]['time'])
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    distri = empty_distri()
    t = 0.0
    #print 'releases: ' + repr(len(releases))
    #print 'states: ' + repr(math.pow(2,len(releases)))
    while (t < deadline):
        i = 0
        #print len(distri)
        job = releases[0]
        while(job[0]['time']== t):
            distri = convolute(distri, job)
            del releases[0]
            if len(releases) > 0:
                job = releases[0]
            else:
                break
            i = i + 1
        if len(releases) > 0:
            t = job[0]['time']
        else:
            t = deadline
        prob = calculate_miss_prob(distri, t)
        probabilties.append(prob)
        times.append(t)
    states.append(len(distri))
    probability = 1
    for i in range(0, len(probabilties),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability


''' The convolution based approach by Maxim and Cucu-Grosjean [17]'''
def convolution(tasks, prob_abnormal, probabilties, states):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    releases = []
    times = []
    calculate_releases(tasks, deadline, releases, prob_abnormal)
    releases = sorted(releases, key=lambda release:release[0]['time'])
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    distri = empty_distri()
    t = 0.0
    print 'releases: ' + repr(len(releases))
    print 'states: ' + repr(math.pow(2,len(releases)))
    while (t < deadline):
        i = 0
        print len(distri)
        job = releases[0]
        while(job[0]['time']== t):
            distri = convolute(distri, job)
            del releases[0]
            if len(releases) > 0:
                job = releases[0]
            else:
                break
            i = i + 1
        if len(releases) > 0:
            t = job[0]['time']
        else:
            t = deadline
        prob = calculate_miss_prob(distri, t)
        probabilties.append(prob)
        times.append(t)
    states.append(len(distri))
    probability = 1
    for i in range(0, len(probabilties),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' The convolution based approach by Maxim and Cucu-Grosjean [17] with state merging'''
def convolution_merge(tasks, prob_abnormal, probabilties, states, pruned):
    tasks = sort(tasks, 'deadline', False)
    deadline = tasks[len(tasks)-1]['deadline']
    releases = []
    times = []
    calculate_releases(tasks, deadline, releases, prob_abnormal)
    releases = sorted(releases, key=lambda release:release[0]['time'])
    states.append(int(math.pow(2,len(releases))))
    min_time = TDA.min_time(tasks, 'execution')
    tasks = sort(tasks, 'execution', True)
    all_times = all_releases(tasks, deadline)
    distri = empty_distri()
    t = 0.0
    while (t < deadline):
        i = 0
        job = releases[0]
        while(job[0]['time']== t):
            distri = convolute(distri, job)
            distri = collapse(distri)
            del releases[0]
            if len(releases) > 0:
                job = releases[0]
            else:
                break
            i = i + 1
        if len(releases) > 0:
            t = job[0]['time']
        else:
            t = deadline
        prob = calculate_miss_prob(distri, t)
        probabilties.append(prob)
        pruned.append(len(distri))
        times.append(t)
    probability = 1
    for i in range(0, len(probabilties),1):
        if (probabilties[i]<probability):
            probability = probabilties[i]
    return probability

''' Calculates the deadline miss probability for a given point in time'''
def calculate_probabiltiy(tasks, time, prob_abnormal, states):
    order = sort(tasks, 'execution', True)
    distributions = []
    # Generates the binomial distribution of the tasks
    for task in order:
        distributions.append(get_distribution(task, time, prob_abnormal))
    # creates an empty distribution as starting point for the convolution
    distri = empty_distri()
    # successively convolutes the starting distribution with the
    for i in range(0,len(distributions),1):
        distri = convolute(distri, distributions[i])
    prob =  calculate_miss_prob(distri, time)
    states.append(len(distri))
    return prob

# probability calculation for one time point with pruning
def calculate_probabiltiy_prune(tasks, time, prob_abnormal, states, pruned_states):
    order = sort(tasks, 'execution', True)
    distributions = []
    for task in order:
        distributions.append(get_distribution(task, time, prob_abnormal))
    min = []
    max = []
    num_states = []
    pruned = []
    prob_cut = []
    prob = 0.0
    minimum = 0.0
    maximum = 0.0
    for distri in distributions:
        min.append(distri[0]['execution'])
        minimum = minimum + distri[0]['execution']
        max.append(distri[len(distri)-1]['execution'])
        maximum = maximum + distri[len(distri)-1]['execution']
    max_states = 1
    if (minimum > time):
        return 1.0
    elif (maximum < time):
        return 0.0
    else:
        distri = empty_distri()
        for i in range(0,len(distributions),1):
            max_states = max_states * len(distributions[i])
            minimum = minimum - min[i]
            maximum = maximum - max[i]
            distri = convolute_prune(distri, distributions[i], minimum, maximum, num_states, pruned, prob_cut, time)
            if (len(distri)>100):
                distri = collapse(distri)
        for i in prob_cut:
            prob = prob + i
        prob = prob + calculate_miss_prob(distri, time)
        max_num_states = np.amax(num_states)
        pruned_states.append(max_num_states)
        states.append(max_states)
        return prob


# probability calculation for one time point with pruning and reduction
def calculate_probabiltiy_prune_reduct(tasks, time, prob_abnormal, states, pruned_states, total_error, max_error_allowed):
    order = sort(tasks, 'execution', True)
    individual_max_error = max_error_allowed / (len(tasks)-1)
    distributions = []
    max_error = []
    i = 0
    for task in order:
        i = i + 1
        error = []
        distri = get_distribution_reduct(task, time, prob_abnormal, error, individual_max_error)
        err = np.sum(error)
        max_error.append(err)
        distributions.append(distri)
    tot_err = np.sum(max_error)
    total_error.append(tot_err)
    min = []
    max = []
    num_states = []
    pruned = []
    prob_cut = []
    prob = 0.0
    minimum = 0.0
    maximum = 0.0
    for distri in distributions:
        min.append(distri[0]['execution'])
        minimum = minimum + distri[0]['execution']
        max.append(distri[len(distri)-1]['execution'])
        maximum = maximum + distri[len(distri)-1]['execution']
    max_states = 1
    if (minimum > time):
        return 1.0
    elif (maximum < time):
        return 0.0
    else:
        distri = empty_distri()
        for i in range(0,len(distributions),1):
            max_states = max_states * len(distributions[i])
            minimum = minimum - min[i]
            maximum = maximum - max[i]
            distri = convolute_prune(distri, distributions[i], minimum, maximum, num_states, pruned, prob_cut, time)
            if (len(distri)>100):
                distri = collapse(distri)
        for i in prob_cut:
            prob = prob + i
        prob = prob + calculate_miss_prob(distri, time)
        max_num_states = np.amax(num_states)
        pruned_states.append(max_num_states)
        states.append(max_states)
        return prob

# calculates the binomial distribution for a given task, time, and probability of abnormal execution
def get_distribution(task, time, prob_abnormal):
    distribution = []
    n = math.ceil(time/task['deadline'])
    for k in range(0, int(n) + 1, 1):
        pair={}
        pair['misses']=k
        pair['prob']= (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*math.pow(prob_abnormal, k)*math.pow((1-prob_abnormal),(n-k))
        pair['execution']=k*task['abnormal_exe']+(n-k)*task['execution']
        distribution.append(pair)
    return distribution

# calculating the reducted distribution for a given task
def get_distribution_reduct(task, time, prob_abnormal, max_error, task_error):
    distribution = []
    n = math.ceil(time/task['deadline'])
    for k in range(0, int(n) + 1, 1):
        pair={}
        pair['misses']=k
        pair['prob']= (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*math.pow(prob_abnormal, k)*math.pow((1-prob_abnormal),(n-k))
        pair['execution']=k*task['abnormal_exe']+(n-k)*task['execution']
        distribution.append(pair)
    # if the original distribution has 6 states or less (5 jobs or less) it is directly kept
    if (len(distribution)<= 6):
        return distribution
    else:
        # otherwise all states are kept (starting from the ones with highest probability)
        # until the total probability of the remaining states is less than the 'task_error'
        total_prob = 0.0
        reducted_distribution = []
        i = 0
        while (i < 6):
            reducted_distribution.append(distribution[0])
            total_prob = total_prob + distribution[0]['prob']
            del distribution[0]
            i = i + 1
        i = 0
        while (len(distribution)> 0 and (total_prob <= 1 - task_error)):
            reducted_distribution.append(distribution[0])
            total_prob = total_prob + distribution[0]['prob']
            del distribution[0]
        if (len(distribution) > 0):
            pair={}
            pair['prob']= 0.0
            while(len(distribution) > 0):
                pair['prob']=pair['prob'] + distribution[0]['prob']
                pair['execution']=distribution[0]['execution']
                if (len(distribution)>1):
                    max_error.append(distribution[0]['prob'])
                del distribution[0]
            reducted_distribution.append(pair)
        return reducted_distribution

# direct convolution of two distributions
def convolute(dist1, dist2):
    dist = []
    for state1 in dist1:
        for state2 in dist2:
            pair={}
            pair['prob']=state1['prob']*state2['prob']
            pair['execution']=state1['execution']+state2['execution']
            dist.append(pair)
    return dist

''' direct convolution of two distributions. The pruning techniques presented in
Section 6.2 are used to prune away unnecessary states. '''
def convolute_prune(dist1, dist2, minimum, maximum, num_states, pruned, prob_cut, time):
    prob = 0.0
    dist = []
    prune = 0
    states = 0
    for state1 in dist1:
        for state2 in dist2:
            states = states + 1
            pair={}
            pair['prob']=state1['prob']*state2['prob']
            pair['execution']=state1['execution']+state2['execution']
            # if a new state will always result in a deadline miss it can be pruned
            # probability of the state is added to the miss probabiltiy
            if ((pair['execution'] + minimum) > time):
                prune = prune + 1
                prob = prob + pair['prob']
            # if a new state will never result in a deadline miss it can be pruned
            elif((pair['execution'] + maximum) < time):
                prune = prune + 1
            # otherwise, it has to be considered further
            else:
                dist.append(pair)
    prob_cut.append(prob)
    pruned.append(prune)
    num_states.append(states)
    return dist

# collapsing the state space by merging states with same workload
def collapse(distri):
    distri = sorted(distri, key=lambda r:r['execution'])
    collapsed = []
    current={}
    current['prob']= 0.0
    current['execution']=distri[0]['execution']
    for state in distri:
        if (current['execution'] == state['execution']):
            current['prob'] = current['prob'] + state['prob']
        else:
            collapsed.append(current)
            current = state
    collapsed.append(state)
    return collapsed

# Calculates the deadline miss probability for a given distribution and the time,
# i.e., tests if the workload is larger than the execution time and sums up the
# related probabilities.
def calculate_miss_prob(distribution, time):
    prob = np.longdouble(0.0)
    for dist in distribution:
        if (dist['execution']>time):
            prob = prob + dist['prob']
    return prob

# calculates the time for the last releases of all tasks before the deadline (and adds the deadline)
# (Binomial based approach)
def last_release(tasks, deadline):
    times = []
    for task in tasks:
        times.append(math.floor(deadline/task['deadline'])*task['deadline'])
    return times

# calculates the time for all releases of all tasks before the deadline (and adds the deadline)
# (Binomial based approach)
def all_releases(tasks, deadline):
    times = []
    times.append(deadline)
    for task in tasks:
        count = task['period']
        while(count < deadline):
            times.append(count)
            count = count + task['period']
    return times

# creates the jobs that have to be convoluted in the convolution based approach
def calculate_releases(tasks, deadline, releases, prob_abnormal):
    for task in tasks:
        time = 0.0
        while(time < deadline):
            distribution = []
            for k in range(0, 2, 1):
                pair={}
                pair['time']=time
                pair['prob']= math.pow(prob_abnormal, k)*math.pow((1-prob_abnormal),(1-k))
                pair['execution']=k*task['abnormal_exe']+(1-k)*task['execution']
                distribution.append(pair)
            releases.append(distribution)
            time = time + task['period']

def sort(tasks, criteria, reverse_order):
    return sorted(tasks, key=lambda item:item[criteria], reverse=reverse_order)

# initializes an empty distribution (workload 0 with probability 1)
def empty_distri():
    distri = []
    pair={}
    pair['misses']=''
    pair['prob']=np.longdouble(1.0)
    pair['execution']=0.0
    distri.append(pair)
    return distri
