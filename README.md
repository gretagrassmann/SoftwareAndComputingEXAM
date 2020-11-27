# Protein interface prediction with Graph Convolutional Networks
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 

## Inclusion of the edges features
To try to improve the [original model](https://github.com/pchanda/Graph_convolution_with_proteins) classification we try to provide for some differentiation between neighbors,
so as to include in the analysis the internal of proteins structures.
The 20 features that characterized each node (amino acid) are now considered in the convolutional operation, with the addition of a tensor ***We*** of dimension
*# features for each edge* * *# edges connected to the central node* in which each element has *# filters* values.
