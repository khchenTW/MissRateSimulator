# Event-based Miss Rate Simulator

# Description of the adopted files from [1]:
- EPST.py contains the .
- bounds.py contains different bounds related to the Chernoff bound.
- task-generator.py contains the task generating routines.
- sort_task_set.py contains the task generating routines.
- cprta.py contains the convolution-based approaches implemented in Python.
- TDA.py contains some time demand analyses.
- timing.py contains some means for time measurement.

# The proposed methods and the simulator files:
- experiments.py contains the main function to run the simulator and the evaluation.
- simulator.py contains the class of the event-based simulator with fault-injection.

# How to use?
- "python experiements.py 0 1 0", Generate task sets. The configuration can be changed at the top of experiments.py.
- "python experiements.py 1 1 0", Quickly get the expected deadline miss rates via the standard display.
- "python experiements.py 2 1 0", Trigger the simulator accordingly and also evaluate the expected miss rates.
- "python experiements.py 3 1 0", Shows the trends of \phi_{k,j}, where j in 1 to 10.

# Experimental setups in the paper:
- TBD
- TBD
- TBD

# Reference
- [1] K. H. Chen and J. J. Chen, "Probabilistic schedulability tests for uniprocessor fixed-priority scheduling under soft errors," 2017 12th IEEE International Symposium on Industrial Embedded Systems (SIES), Toulouse, France, 2017, pp. 1-8.
