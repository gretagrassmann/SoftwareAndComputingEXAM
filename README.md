# Protein interface prediction with Graph Convolutional Networks
## Software and Computing for Applied Physics exam with Prof. Giampieri, University of Bologna 

## Configuration of a customized model
To allow the user to explore more easily different configurations of the model's architecture and execution, all the parameters that define the network can be changed as desired, with a few limitations that are now going to be explained. The values now specified in **configuration.txt** are the ones used in the **master branch**, as well as the ones that according to the [original paper](https://www.semanticscholar.org/paper/Protein-Interface-Prediction-using-Graph-Networks-Fout-Byrd/c751ab01aedc2888a7fe6e8b4f77ab1afa94072f) give the best results. Nevertheless new values for the following parameters can be tested:
* ***number_of_epochs***: the training is executed for a number of epochs going from 1 to this value.
* ***number_of_residues_couples_for_minibatch***: each minibatch contains this number of paired ligand and receptor residues. This value must be smaller than the lower total number of pairings (***n  = len(pair_examples)***).
* ***results_dropout_probability***: probability with which the new values are the obtained one, scaled up by 1/this value. Otherwise the output is 0.
* ***number_filters***: number of filters used in the convolutional layers.
* ***selected_models_for_testing***: the models resulting from the training with the here specified number of epochs are tested. The minimum value that can be inserted is 0, the maximum value is ***number_of_epochs***-1.
* ***number_of_convolutional_layers***: number of convolutional layers.
* ***convolutional_layers_activation_function***: defines the activation function of the convolutional layers. With ***"ReLU_function"*** a ReLU activation function is selected, while with ***"tanh_function"*** a hyperbolic function is used for the activation. Otherwise the activation is given by a linear function.
* ***relation_between_convolutional_branches_weights***: determines if the two branches of the convolutional part of the architecture (one for the ligand and one for the receptor residues) share the same weights. This is the case when ***"shared"*** is selected, otherwise the two branches will have different weights.
* ***edges_features***: determines if the edges' features are considered in the convolution operation. Only two insertion are accepted: with ***"yes"*** the edges are taken into account with a new weight matrix ***We***, instead with ***"no"*** they are not considered.
* ***type_of_train_data***: with ***"DBD"*** the data from the Docking Benchmark Dataset (DBD) are used, otherwise the user has to specify the new data file by inserting its path. The type of data that can be used are [here](https://github.com/gretagrassmann/SoftwareAndComputingEXAM/tree/master#dataset) specified.
* ***type_of_test_data***: with ***"DBD"*** the data from the Docking Benchmark Dataset (DBD) are used, otherwise the user has to specify the new data file by inserting its path. The type of data that can be used are [here](https://github.com/gretagrassmann/SoftwareAndComputingEXAM/tree/master#dataset) specified.







