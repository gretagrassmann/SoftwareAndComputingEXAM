# Protein interface prediciton with Graph Convolutional Networks
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 

### Table of contents:
  * [A typical structure of the Graph Neural Network model: the Graph Convolutional Network](#a-typical-structure-of-the-graph-neural-network-model-the-graph-convolutional-network)
  * [Graph Convolutional Networks for protein-protein interaction](#graph-convolutional-networks-for-protein-protein-interaction)
  * [Implementation of the basic model](#implementation-of-the-basic-model)
  * [Dataset](#dataset)
  * [Strucuture of our project](#strucuture-of-our-project)
    + [Master Branch](#master-branch)
      - [graph_conv.py](#graph_convpy)
      - [train.py](#trainpy)
      - [test.py](#testpy)
    + [EdgesFeatures Branch](#edgesfeatures-branch)
    + [IndependentParameters branch](#independentparameters-branch)
    + [TransitionFunction branch](#transitionfunction-branch)
  * [Running the experiment](#running-the-experiment)
    + [Requirements](#requirements)
- [CONTACTS](#contacts)




### A typical structure of the Graph Neural Network model: the Graph Convolutional Network
The Graph Neural Network (**GNN**) is a connectionist neural network model which has gained increasing popularity in various domains related to graph analysis, since many underlying relationships among data in several areas of science can be naturally represented in terms of graph structures. Its success is due to its power in modeling the dependencies between nodes in a graph by capturing the messages passing between them. In a simplified approach each node of the graph is described by a set of features, and the network uses the features from each node and its neighbouring ones to infer a state embedding which contains the information of the neighborhood of each node. In the more complete original approach this information propagation between nodes is guided by the edges.<br /> 
Several modifications of **GNNs** have been proposed, each of them with different aggregator functions to gather information from the neighbors of a node and specific updaters.
One of them are the Graph Convolutional Networks (**GCNs**), which learn to integrate node features based on labels and link structures, by generalizing the standard notion of convolution over a regular grid (representing a sequence or an image) to convolution over a graph structures. They can be thought of as fully-connected neural networks plus a neighborhood aggregation component. The former computes a non-linear feature projection, whereas the latter mixes the feature of each node with those of its neighbors. The objective is to design a convolutional operator that can be applied to graph without a regular structure, without imposing a particular order on the neighbors of a given node. The convolution is performed by the convolutional layers for which a number of filters must be specified. The filters are what actually detects specific patterns; the deeper the network goes, the more sophisticated patterns these filters can pick-up. To make useful predictions, the **GCN** will need to make use of other kinds of layers such as dense, pooling layer structures. <br /> 

### Graph Convolutional Networks for protein-protein interaction
**GCNs** can be applied on structured objects that can naturally be modeled as graphs, for example for protein interface prediction, a problem described [here](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f).
The resulting structure information is useful for predicting a variety of property of proteins, including their function and interaction with other proteins or with DNA or RNA. <br />
Each residue (amino acid) in a protein is a node in a graph. The neighborhood of that node used in the convolutional operator is the set of *k* closest residues as determined by the mean distance between their atoms in the protein structure. The following Figure shows a schematic description of the convolution operator which has as receptive field a set of neighboring residues, and produces an activation which is associated with the center residue, that we are going to call **center node**.
<img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/protein.PNG" alt="Drawing" width = "500" alt=""></img><br />
Each node has features computed from its amino acid sequence and structure. Edges have features describing for example the relative distance and angle between residues. 
By performing a convolution over the local neighborhood of a node and stacking multiple layers of convolution we can learn latent representations that integrate information across the graph that represent the three dimensional structure of a protein of interest. Then a follow-up neural network architecture combines this learned features across pairs of proteins and classifies pairs of amino acid residues as part of an interface or not.
A possible convolutional operator could be defined so as to provide the following output of a set of filters in a neighborhood *ne[n]* of a **center node** of interest *n*: 

<img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/Convolution_withoud_edges.PNG" alt="Drawing" width = "300" alt=""></img><br />

Where *Wc* is the weight matrix associated with the **center node**, *Wn* the one associated to its neigbors, *b* is a vector of biases and *σ* is non-linear activation function. This is the operation performed in the [original code](https://github.com/pchanda/Graph_convolution_with_proteins.git), on which the codes presented in this repository are based. 

### Implementation of the basic model
To solve the protein interface prediction problem we need to classify pairs of nodes from two separate graph, representing the ligand and the receptor protein. The examples are composed of pairs of residues, one from each one of them: the data are a set of labeled pairs of amino acid from the ligand and receptor protein with the associated label that indicates if the two are interacting or not. <br />
As schematically shown in the following Figure, each neighborhood of a residue in the two proteins is processed using two graph convolutional layers, that learn their feature representations. The weights of the two "legs" (the two convolutional branches of the whole network, one dedicated to ligands, the other to receptoris) of this network architecture are shared.
The activations generated by the convolutional layers are merged by concatenating them in a representation of residue pairs: since the role of ligand and receptor is arbitrary, the scoring function should be learned independently of the order in which the two residues are presented to the network.
Finally these resulting features are passed through two regular dense layers before classification. The classification is performed on an average of the outputs of the fully-connected dense layers for the ligand-receptor and receptor-ligan pairs.

<img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/original.png" alt="Drawing" width = "300" alt=""></img><br />

Each protein is split into minibatches of 128 paired examples, and 256 filters are applied. These values, so as all the following constant parameters (like the number of dense layer chosen as two or the negative to positive example ratio as 1:10) were inspired by the result of a validation set performed by the authors of [this paper](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f).

### Dataset
The studied data, automatically downloaded by our code from [here](https://github.com/pchanda/Graph_convolution_with_proteins/tree/master/data) for simplicity, but originally [here presented](https://zenodo.org/record/1127774#.X70Z7uWSk2y) are derived from protein complexes in the Docking Benchmark Dataset (DBD) version 5.0, which is the standard benchmark dataset for assessing docking and interface prediction method. These complexes are selected subsets of structures from the Protein Data Bank (PDB). The PDB is a database for the three dimensional structural data of large biological molecules, such as proteins and nucleic acids, typically obtained by X-ray crystallography, Nuclear Magnetic Resonance (NMR) spectroscopy, or increasingly cryo-electron microscopy.
Because in any given complex the majority of residue pairs do not interact, the negative examples were downsized in the training dataset to obtain a 10:1 ratio of negative and positive examples. Positive examples are residue pairs that participate in the interface, negative examples are pairs that do not.
The number of neighbors is fixed in the dataset as 20.

### Strucuture of our project
The following main codes were developed by [Lorenzo Spagnoli](https://github.com/LorenzoSpag) and Greta Grassmann. The objective of our work is improving the prediciton of interfaces between protein residues by bettering the simple graph convolutional deep learning method performed in the [original code](https://github.com/pchanda/Graph_convolution_with_proteins.git). <br />
The original code implements a basic approach that does not consider the internal structure of the protein, in that it neglects the edges' features, so we tried to improve its classification precision. As already mentioned, the values of the hyperparameters are inspired by a preceding work, and in our study no extensive search over all the space of possible feature representation and model hyperparameters are done. <br />
However, we analized three particular cases, each one developed inside one of the branches of [this repository](https://github.com/gretagrassmann/SoftwareAndComputingEXAM):
* The difference between a model which includes the edges'features and one that does not.
* The difference between the implementation of a *tanh* function in place of a ReLU as activation function for the convolutional layers.
* The difference between an architecture in which the two "legs" of the network share the weights and one in which they do not.

**avg_loss_plot.py** and **avg_loss_plot_test.py** are used to compare the average loss for increasing number of epochs (for the training and the testing respectively) of the original model and one of our new implementations. <br />
In the following, a description of the organization of the mentioned branches of this repository. A more in depth discussion about the developed softwares, together with a more in depth study of the proposed problem can be found in our [project report](https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf) (written for the Complex Networks exam, with Prof. Remondini).


#### Master Branch
In this branch the original code is presented, with some slight modifications. In particular, from now on the data files will be automatically downloaded from the code, and the libraries are adapted to a more recent version of Python.

##### graph_conv.py
The placeholder tensors for building the graph convolutional network are defined. The network architecture here defined is the one already desribed in the Sections **Graph Convolutional Networks for protein-protein interaction** and **Implementation of the basic model** : only the features of the nodes are used and the "legs" share the same weights. The activation function of the convolutional layers is a ReLU. <br />
The weights matrices are tensors of dimension *#features* × *#filters* elements (remeber that they will have to multiply the data presented in a tensor with dimension *#nodes* × *#features*). The two dense layers are defined as a square matrix *2·#filters* × *2·#filters* and another one with size *2·#filters* × *1*.  <br />
There is also the definition of the unification of the two matrices of dimension *minibatch size* × *#filters* given as ouput by the convolutional layers in a single matrix
of dimension the *2·minibatch size* × *2·#filters*. In this way both the ligand-receptor and receptor-ligand pairs are considered. A function which performs the final average for the classification of the outputs of the fully-connected layer for the ligand-receptor and receptor-ligand pairs is defined too.

#####  train.py
The model defined in **graph_conv.py** is trained on 175 pairs of proteins for the desired number of epochs. Ligand end receptor residues are fed separately into the two ”legs” of the network’s architecture deined in **graph_conv.py**.  <br />
For each epoch, the average loss and the resulting parameter of the model are saved.

##### test.py
The model with the parameters corresponding to the selected numbers of epochs is tested on 55 pairs of proteins. For each model, the average loss and the area under the ROC curve are saved.

#### EdgesFeatures Branch
To understand if the original model prediction can be improved, we have added a matrix *We* that takes into consideration the edges features, so that now the convolutional operator has this form:

<img src="https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/master/images/Convolution_with_edges.PNG" alt="Drawing" width = "400" alt=""></img><br />

To apply this modification we had to manipulate **graph_conv.py**. The other codes were changed only in that the name of the written are read files are adapted.

#### IndependentParameters branch
We test if the implementation of non-shared weights between the two ”legs” of the architecture leads to an improvement in the model predictions. Indeed the two ”legs” process different kind of proteins, respectively a receptor and a ligand. Since they have different functioning, it would makes sense for them to have different morphology and as a consequence for the corresponding graphs to have different weights. <br />
As before, to apply this modification we had to manipulate **graph_conv.py**. The other codes were changed only in that the name of the written are read files are adapted.

#### TransitionFunction branch
We swap the ReLU activation function σ with a tanh function in the two convolutional layers. The ReLU activation function is still implemented in the dense layers, since the output on which we want to perform the classification has to be a positive number.
As before, to apply this modification we had to manipulate **graph_conv.py**. The other codes were changed only in that the name of the written are read files are adapted.

### Running the experiment
#### Requirements 
* Python 3.7
* numpy 1.18.1
* tensorflow 1.14.0

All the files that have to be read (like the data files **train.txt** and **test.txt**) or written (like the average loss for the training and testing results) are saved in the folder in which the codes are downloaded. <br />
Because of the long computational time required for both training and testing, the results for each of the four studied models are already at disposal in [Experiment_running_RESULTS](https://github.com/gretagrassmann/SoftwareAndComputingEXAM/tree/master/Experiment_running_RESULTS). This folder contains for each model the average loss of the training for increasing number of epochs (from 1 to 149) and the average loss together with the area under the ROC curveof the testing for the models trained with 1,11,51,101 and 149 epochs.<br />
Otherwise, for each branch the procedure to follow is the same:
1. Run **graph_conv.py**, which defines the model that is going to be implemented and downloads the test and train data. If desired the number of epochs (***num_epochs***) can be changed. 
2. Run **train.py**, which takes the data in **train.txt** and train the selected model for the desired number of epochs. For each number of epochs the average loss and the model's parameters are saved.
3. Run **test.py**, which takes the models correponding to the ones trained for a number of epochs selected in ***n=[]*** and tests them on the data in **test.txt**. The average loss and the area under the ROC curve for each one of them are saved.

The classification precision of two models can then be compared with **avg_loss_plot.py** and  **avg_loss_plot_test.py**.



## CONTACTS
Please direct any question to:
  * gretagrassmann0@gmail.com
  * Lorenzo.rspagnoli@gmail.com
