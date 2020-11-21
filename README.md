# SoftwareAndComputingEXAM
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 
These codes are based on the original work presented [here](https://github.com/pchanda/Graph_convolution_with_proteins.git) to predict interfaces between protein residues by
implementing a simple graph convolutional deep learning method. The theory and a more in-depth approach to classify a pair of amino acid residues as interacting or not are
discussed [here](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f).

The following main codes were developed by [Lorenzo Spagnoli](https://github.com/LorenzoSpag) and Greta Grassmann.

### graph_conv.py
The placeholder tensors for building the graph convolutional network are defined. The network is the one used 
[here](https://github.com/pchanda/Graph_convolution_with_proteins.git): only the features of the nodes are used, the transition function of the convolutional layers is a ReLU
and the weights are shared between the two "legs" of the network. The codes implement the following network architecture:
<pre>
ligand-residue->first convolutional layer->second convolutional layer<br /> 
                                                                     | <br />
                                                                      ->merge layer->first dense layer->second dense layer->prediciton <br />
                                                                     | <br />
 residue-ligand->first convolutional layer->second convolutional layer          
 </pre>                                                               

### train.py
The model defined in graph_conv.py is trained on 175 pairs of proteins for the desired number of epochs. For each epoch, the average loss and the resulting parameter of the model are saved.

### test.py
The model with the parameters corresponding to the selected numbers of epochs is tested on 55 pairs of proteins. For each model, the average loss and the ROC curve value are saved.

## CONTACTS
Please direct any question to:
  * gretagrassmann0@gmail.com
  * Lorenzo.rspagnoli@gmail.com
