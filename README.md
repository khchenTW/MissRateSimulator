# Event-based Miss Rate Simulator and Deadline Miss Rate

# Environment:
- Python 2.7

# Description of the adopted files from [1] and [2]:
- EPST.py contains the analyses of the upper bound of the deadline misses.
- bounds.py contains different bounds related to the Chernoff bound.
- task-generator.py contains the task generating routines (it is enhanced by [2] files).
- sort_task_set.py contains the task generating routines.
- TDA.py contains some time demand analyses (it is enhanced by [2] files).
- deadline_miss_probability.py contains the methods from [2].

# The proposed methods and the simulator files:
- experiments.py contains the main function to run the simulator and the evaluations.
- simulator.py contains the class of the event-based simulator.

# How to use? (10 tasks with 1 set)
- "python experiements.py 0 10 1 0", Generate task sets. The configuration can be changed at the top of experiments.py.
- "python experiements.py 1 10 1 0", Quickly get the expected deadline miss rates via the standard display.
- "python experiements.py 2 10 1 0", Trigger the simulator accordingly and also evaluate the expected miss rates.
- "python experiements.py 3 10 1 0", Shows the trends of \phi_{k,j}, where j in 1 to 10.
- "python experiements.py 4 10 1 0", Shows the motivational example for the differences between the deadline miss rate and the probability deadline misses.

# Experimental setups in the paper:
- All the results are statically stored in each ploter in "printer" folder
- To obtain Figure 1, run "python experiements.py 4 2 1 0"
- To obtain Figure 4, run "python experiements.py 3 5 5 0"
- To obtain Figure 5, run "python experiements.py 2 2 30 0"
- To obtain Figure 6, run "python experiements.py 1 10 100 0"

# Future work / Pending Feature for the simulator:
- dynamic-priority scheduling policies
- non-preemptive task systems

# Reference
- [1] K. H. Chen and J. J. Chen, "Probabilistic schedulability tests for uniprocessor fixed-priority scheduling under soft errors", 2017 12th IEEE International Symposium on Industrial Embedded Systems (SIES), Toulouse, France, 2017, pp. 1-8.
- [2] Georg von der Brüggen, Nico Piatkowski, Kuan-Hsun Chen, Jian-Jia Chen, Katharina Morik, "
Efficiently Approximating the Probability of Deadline Misses in Real-Time Systems", accepted in ECRTS 2018.
