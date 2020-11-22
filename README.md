# Protein interface prediciton with Graph Convolutional Networks
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 

### A typical structure of the Graph Neural Network model: the Graph Convolutional Network
The Graph Neural Network (**GNN**) is a connectionist neural network model which has gained increasing popularity in various domains related to graph analysis, since many underlying relationships among data in several areas of science can be naturally represented in terms of graph structures. Its success is due to its power in modeling the dependencies between nodes in a graph by capturing the messages passing between them. In a simplified approach each node of the graph is described by a set of features, and the netwrok uses the features from each node and its neighbouring ones to infer a state embedding which contains the information of the neighborhood of each node. In the more complete original approach this information propagation between nodes is guided by the edges.<br /> 
Several modifications of **GNNs** have been proposed, each of them with different aggregator functions to gather information from the neighbors of a node and specific updaters.
One of them are the Graph Convolutional Networks (**GCNs**), which learn to integrate node features based on labels and link structures, by generalizing the standard notion of convolution over a regular grid (representing a sequence or an image) to convolution over a graph structures. They can be thought of as fully-connected neural networks plus a neighborhood aggregation component. The former computes a non-linear feature projection, whereas the latter mixes the feature of each node with those of its neighbors. The objective is to design a convolutional operator that can be applied to graph without a regular structure, without imposing a particular order on the neighbors of a given node. The convolution is performed by the convolutional layers for which a number of filters must be specified. The filters are what actually detects specific patterns; the deeper the network goes, the more sophisticated patterns these filters can pick-up. To make useful predictions, the **GCN** will need to make use of other kinds of layers such as dense, pooling layer structures. <br /> 

### Graph Convolutional Networks for protein-protein interaction
**GCNs** can be applied on structured objects that can naturally be modeled as graphs, for example for protein interface prediction, a problem described [here](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f).
The resulting structure information is useful for predicting a variety of property of proteins, including their function and interaction with other proteins or with DNA or RNA. <br />
Each residue (amino acid) in a protein is a node in a graph. The neighborhood of that node used in the convolutional operator is the set of *k* closest residues as determined by the mean distance between their atoms in the protein structure. The following Figure shows a schematic description of the convolution operator which has as receptive field a set of neighboring residues, and produces an activation which is associated with the center residue, that we are going to call **center node**:.
<p >
  <img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/protein.PNG" / width="500" class="img-left" alt="">
</p>
Each node has features computed from its amino acid sequence and structure. Edges have features describing for example the relative distance and angle between residues. 
By performing a convolution over the local neighborhood of a node and stacking multiple layers of convolution we can learn latent representations that integrate information across the graph that represent the three dimensional structure of a protein of interest. Then a follow-up neural network architecture combines this learned features across pairs of proteins and classifies pairs of amino acid residues as part of an interface or not.
A possible convolutional operator could be defined so as to provide the following output of a set of filters in a neighborhood *ne[n]* of a **center node** of interest *n*: 
<p >
  <img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/Convolution_withoud_edges.PNG" / width="300" class="img-left" alt="">
</p>
Where *Wc* is the weight matrix associated with the **center node**, *Wn* the one associated to its neigbors, *b* is a vector of biases and *σ* is non-linear activation function. This is the operation performed in the [original code](https://github.com/pchanda/Graph_convolution_with_proteins.git), on which the codes presented in this repository are based. 

### Implementation of the basic model
To solve the protein interface prediction problem we need to classify pairs of nodes from two separate graph representing the ligand and the receptor protein. The examples are composed of pairs of proteins, one from each one of them: the data are a set of labeled pairs of ligand and receptor protein with the associated label that indicates if the two are interacting or not. <br />
As schematically shown in the following Figure, each neighborhood of a residue in the two proteins is processed using two graph convolutional layers, that learn their feature representations. The weights of the two "legs" of this network architecture are shared.
The activations generated by the convolutional layers are merged by concatenating them in a representation of residue pairs: since the role of ligand and receptor is arbitrary, the scoring function should be learned independently of the order in which the two residues are presented to the network.
Finally these resulting features are passed through two regular dense layers before classification. The classification is performed on an average of the outputs of the fully-connected dense layers for the ligand-receptor and receptor-ligan pairs.
<p >
  <img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/original.png" / width="300"  alt="">
</p>
Each protein is split into minibatches of 128 paired examples, and 256 filters are applied. These values, so as all the following constant parameters (like the number of dense layer chosen as two or the negative to positive example ratio as 1:10) were inspired by the result of a validation set performed by the authors of [this paper](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f).

### Dataset
These data are derived from protein complexes in the Docking Benchmark Dataset (DBD) version 5.0, which is the standard benchmark dataset for assessing docking and interface prediction method. These complexes are selected subsets of structures from the Protein Data Bank (PDB). The PDB is a database for the three dimensional structural data of large biological molecules, such as proteins and nucleic acids, typically obtained by X-ray crystallography, Nuclear Magnetic Resonance (NMR) spectroscopy, or increasingly cryo-electron microscopy.
Because in any given complex the majority of residue pairs do not interact, the negative examples were downsized in the training dataset to obtain a 10:1 ratio of negative and positive examples. Positive examples are residue pairs that participate in the interface, negative examples are pairs that do not.
The number of neighbora is fixed in the dataset as 20.

### New software development
The following main codes were developed by [Lorenzo Spagnoli](https://github.com/LorenzoSpag) and Greta Grassmann. The objective of our work is improving the prediciton of interfaces between protein residues by bettering the simple graph convolutional deep learning method performed in the [original code](https://github.com/pchanda/Graph_convolution_with_proteins.git). <br />
**avg_loss_plot.py** and **avg_loss_plot_testing.py** are used to compare the average loss for increasing number of epochs (for the training and the testing respectively) of the original model and one of our new implementations. <br />


#### Master- graph_conv.py
The placeholder tensors for building the graph convolutional network are defined. The network is the one used 
[here](https://github.com/pchanda/Graph_convolution_with_proteins.git): only the features of the nodes are used, the transition function of the convolutional layers is a ReLU
and the weights are shared between the two "legs" of the network.                                            

#### Master- train.py
The model defined in graph_conv.py is trained on 175 pairs of proteins for the desired number of epochs. For each epoch, the average loss and the resulting parameter of the model are saved.

#### Master- test.py
The model with the parameters corresponding to the selected numbers of epochs is tested on 55 pairs of proteins. For each model, the average loss and the ROC curve value are saved.

## CONTACTS
Please direct any question to:
  * gretagrassmann0@gmail.com
  * Lorenzo.rspagnoli@gmail.com
