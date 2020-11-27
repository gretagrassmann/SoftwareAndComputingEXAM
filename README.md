# Protein interface prediction with Graph Convolutional Networks
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 

## Separation of the weights in the two "legs" of the architecture

To try to improve the [original model](https://github.com/pchanda/Graph_convolution_with_proteins) classification, we got rid of the constriction imposed on the weights by
the original code: while in that case the weights of the convolutional layers had to be the same for the processing of the ligand-receptor and receptor-ligand pairs, in this
new code they are allowed to assume different values during the training.
