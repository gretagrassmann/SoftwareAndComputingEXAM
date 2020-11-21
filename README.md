# SoftwareAndComputingEXAM
Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 
## New transition function for the convolutional layers: tanh in place of ReLU

To try to improve the prediciton of the [original model](https://github.com/pchanda/Graph_convolution_with_proteins) we try changing
the transition function from a ReLu to a tanh. Looking at the plots produced in **avg_loss_plot.py** and **avg_loss_plot_testing.py** it can be seen 
that the ReLU function is indeed the one with better results.
