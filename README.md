
# Event-based Miss Rate Simulator and Deadline Miss Rate

# Environment:
- Python 3.6

# Description of the adopted files from [1] and [2]:
- EPST.py contains the analyses of the upper bound of the deadline misses.
- bounds.py contains different bounds related to the Chernoff bound.
- task-generator.py contains the task generating routines (it is enhanced by [2] files).
- sort_task_set.py contains the task generating routines.
- TDA.py contains some time demand analyses (it is enhanced by [2] files).
- deadline_miss_probability.py contains the methods from [2].

# The proposed methods and the simulator files:
- experiments.py contains the main function to run the simulator and the evaluations.
- multiprocessor_simulator.py / simulator.py contains the class of the event-based simulator.

# How to use?
To run the simulator, run the following command from the repository:

```python3 experiments.py [mode] [# tasks] [tasksets_amount] [generationType] [part]```

## mode:
0: generates tasksets using the other configuration parameters  
2: starts a simulation with all combinations of the current configuration and saves the miss rate for each in the outputs folder  
3: creates a plot for the current configuration  
4: takes the current configuration and calculates the average amount of releases of all tasks for the generated taskset required, to reach the specified jobnumber  
7: for testing purposes

## \# tasks:
desired number of tasks

## tasksets_amount:
desired number of tasksets

## generationType:
0: preset of possible periods (1, 2, 5, 10, 50, 100, 250 and 1000)  
1: random periods between 1 and 100

## part:
allows saving different tasksets with identical configuration

## in-file configurations:
The scheduling method and processor type have to be set in experiments.py (lists 'schedulingMethods' and 'processorTypes').
### processorType:
0: single  
1: partitioned  
2: global  

### scheduling:
1: random priority  
2: earliest deadline first  
3: deadline-monotonic  
4: rate-monotonic  


# Reference
- [1] K. H. Chen and J. J. Chen, "Probabilistic schedulability tests for uniprocessor fixed-priority scheduling under soft errors", 2017 12th IEEE International Symposium on Industrial Embedded Systems (SIES), Toulouse, France, 2017, pp. 1-8.
- [2] Georg von der Brüggen, Nico Piatkowski, Kuan-Hsun Chen, Jian-Jia Chen, Katharina Morik, "
Efficiently Approximating the Probability of Deadline Misses in Real-Time Systems", accepted in ECRTS 2018.
