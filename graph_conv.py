"""
A more in depth explanation of the theory behind this model can be found at
https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf.
"""
import tensorflow as tf
import numpy as np
import re
from urllib import request
import gzip
import shutil

# The model will be trained for a #epochs between 1 and num_epochs
num_epochs = 150
# Number of paired examples in which the protein is divided
minibatch_size = 128
dropout_keep = 0.5

def initializer(init, shape):
    """This function initializes the weights' values in the convolutional and dense layers, so as the vector of biases
           b.
          Parameters:
              init : string that determines if the weights are zero-valued (init == "zero") or extracted from a
                     uniform distribution (init == "he").
              shape : dimension of the input tensor that has to be initialized.
          Returns:
              The initial values of the weights, from which the training can start."""
    if init == "zero":
        # The generated tensor has only zero-valued elements
        return tf.zeros(shape)
    elif init == "he":
        # fan_in gives the number of element in the tensor
        fan_in = np.prod(shape[0:-1])
        std = 1 / np.sqrt(fan_in)
        # The generated values follow a uniform distribution in the range [-stt,std)
        return tf.random_uniform(shape, minval=-std, maxval=std)

def nonlinearity(nl):
    """This function determines what kind of activation function is implemented in the convolutional and
          dense layers.
             Parameters:
                nl : string that determines if the activation function is a ReLU (nl == "relu"), a
                     tanh function (nl == "tanh") or a linear one (nl == "linear").
             Returns:
                 The function object determined by nl."""
    if nl == "relu":
        return tf.nn.relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x

def node_average_model(input, params, filters=None, dropout_keep_prob=1.0, trainable=True):
    """This function defines the operations performed in the convolutional layers: starting from some initial values for
          the weights and the features of the nodes and of the edges, they multiply them. In particular, there are three weights
          tensors, one that multiplies the center node features (Wc), one that multiplies the neighbors features (Wn), and
          one that multiply the edges' features (We): by taking into account the information of the different edges
          between each center node and its neighbors, we can provide some differentiation between neighbors. 
          There is a vector of bias (b) too.
          The objective of the convolution is to aggregate the information from the neighborhood of
          a node to model the dependencies between the nodes belonging to a graph (in this case the amino acid forming a
          protein): it infers a state embedding which contains the information of the neighborhood of each node. A number
          of filters must be specified for the convolution: the filters are what actually detects specific patterns. At the
          ond to each element a value for each applied filter is associated.
          A more in depth description of the following operations is given at
          https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf.
             Parameters:
                 input : tensor with the features of the edges and vertices related to the ligand or receptor residue, and
                         the definition of the neighborhood of each node.
                 params : values of the weights for which these features are multiplied.
                 filters : number of considered filters. The weights will be tensors of dimension [#features,#filters].
                 dropout_keep_prob : probability with which the new features are the obtained one, scaled
                                     up by 1 / dropout_keep_prob. Otherwise the output is 0.
                 trainable : string that determines if the variable can be acted upon by the optimizer.
             Returns:
                 z : new features of edges and vertices. To each one of them a different value for each one of the applied
                     filters is associated.
                 params : new weights"""
    # The input tensor is divided in features referred to edges and vertices and in the definition of the neighborhood of
    # each node
    vertices, edges, nh_indices = input
    # nh_indices goes from shape [minibatch size, #neighbors,1] to [minibatch size, #neighbors]
    nh_indices = tf.squeeze(nh_indices, axis=2)
    # vertices has shape [#nodes,#features]
    v_shape = vertices.get_shape()
    e_shape = edges.get_shape()
    # For fixed number of neighbors, -1 is a pad value
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32),
                              -1)
    if params is None:
        # Creates new weights if the convolutional layer has just been entered
        # Wc is the tensor of weight referred to the features of the center node. It has shape [#features,#filters]
        Wvc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvc",
                          trainable=trainable)
        # bv is the vector of bias. It has dimension [#filters]
        bv = tf.Variable(initializer("zero", (filters,)), name="bv", trainable=trainable)
        # Wn is the tensor of weight referred to the features of the neighbors nodes. It has shape [#features,#filters]
        Wvn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvn",
                          trainable=trainable)
        # This matrix considers the edges' features
        We = tf.Variable(initializer("he", (e_shape[2].value, filters)), name="We",
                         trainable=trainable)
    else:
        # Takes as weights and vector of bias the ones resulting from the preceding cycle
        Wvn, We = params["Wvn"], params["We"]
        Wvc = params["Wvc"]
        bv = params["bv"]
        filters = Wvc.get_shape()[-1].value
    params = {"Wvn": Wvn, "We": We, "Wvc": Wvc, "bv": bv}
    # Generates center node signals
    Zc = tf.matmul(vertices, Wvc, name="Zc")  # [#vertices,#filters]
    # Creates neighbors signals
    e_We = tf.tensordot(edges, We, axes=[[2], [0]], name="e_We")  # (n_verts, n_nbors, filters)
    v_Wvn = tf.matmul(vertices, Wvn, name="v_Wvn")  # (n_verts, filters)
    # tf.gather reorders the neighbors signals according to their position in the neighborhood.
    # Zn is a tensor of shape [#vertices,#filters] with the normalized sum, for each vertex and each filter, of that
    # neighborhood's vertices signals
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wvn, nh_indices), 1) + tf.reduce_sum(e_We, 1),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))
    # Activation function
    nonlin = nonlinearity("relu")
    # Sum of the signals from the center node itself and its neighbors, plus the vector of bias
    sig = Zn + Zc + bv
    # To each node a value is associated for each one of the applied filters
    z = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, dropout_keep_prob)
    return z, params

def dense(input, out_dims=None, dropout_keep_prob=1.0, nonlin=True, trainable=True):
    """This function defines the operations performed in the dense layers. Starting from some initial values for
       the residue, whose number depends on how many filters are considered, it combines the values corresponding to
       the different filters to assign a lower number of values to each residue.
          Parameters:
              input : tensor with a certain starting number of values for each residue.
              out_dims : number of values that each residue will have at the end of the layer.
              dropout_keep_prob : probability with which the new values are the obtained one, scaled
                                  up by 1 / dropout_keep_prob. Otherwise the output is 0.
              nonlin : string that signals that one of the non-linear function defined in function
                       nonlinearity could be used.
              trainable : string that determines if the variable can be acted upon by the optimizer.
          Returns:
              Z : tensor with out_dims number of values for each residue."""
    input = tf.nn.dropout(input, dropout_keep_prob)
    # Initial number of values for each residue
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    # Weights of the dense layer
    W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
    b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
    # Operation performed in the dense layer
    Z = tf.matmul(input, W) + b
    if (nonlin):
        nonlin = nonlinearity("relu")
        Z = nonlin(Z)
    Z = tf.nn.dropout(Z, dropout_keep_prob)
    return Z

def merge(input):
    """This function unifies the output (of dimesion [minibatch size,#filter]) of the two convolutional
       branches of the whole network, one dedicated to the convolution of ligands (1), and the other to
       receptors (2). The activations generated by the convolutional layers are merged by concatenating
       them in a representation of residue pairs: since the role of ligand and receptor is arbitrary, the
       scoring function should be learned independently of the order in which the two residues
       are presented to the network.
              Parameters:
                  input : the output of the two convolutional branches (#filters value for each element) and
                          the parings between the ligand and receptor pairs.
              Returns:
                  A tensor with the output of the convolutional layer in the combinations ligand-receptor
                  and receptor ligand for each pair."""
    input1, input2, examples = input
    out1 = tf.gather(input1, examples[:, 0])
    out2 = tf.gather(input2, examples[:, 1])
    # Ligand-receptor pairs
    output1 = tf.concat([out1, out2], axis=0)
    # Receptor-ligand pairs
    output2 = tf.concat([out2, out1], axis=0)
    return tf.concat((output1, output2), axis=1)

def average_predictions(input):
    """This function performs an average of the outputs of the fully-connected dense layers
       for the ligand-receptor and receptor-ligand pair. The classification will be performed on this average.
                  Parameters:
                      input : the tensor given as output from the last dense layers, which has a single value
                              for each possible combination of the residues pairs.
                  Returns:
                      combined : a tensor with half the dimension of the input, where each element is given by
                                 the mean of two elements of the input corresponding to the same residues pair (in
                                 the two possible combinations)."""
    combined = tf.reduce_mean(tf.stack(tf.split(input, 2)), 0)
    return combined


def build_feed_dict(model_variables_list, minibatch):
    """This function defines what the first convolutional layers (left and right branches) will receive as input
         at each epoch: each protein is divided in minibatches, and to each minibatch three main tensors are
         associated, together with the pairings between the residues from the ligand and receptor minibatch, their associated
         label (interacting or not pairs) and the probability with which the results are discharged.
                    Parameters:
                        model_variables_list : list of all the variables that determine the models, including
                                               the vertices' and edges' features and the definition of the neighborhoods
                                               for both the ligand and the receptor protein, the pairings between their
                                               respective residues, the probability to accept the results and the correct
                                               label for each pair.
                        minibatch : considered portion of the protein, with all the relative features and characteristic.
                    Returns:
                        feed_dict : Input tensor that is going to be fed to the architecture.
                                    It includes : in_vertex1 : tensor of shape [minibatch size, #vertex features] that
                                                               contains the vertices' features of the ligand protein.
                                                  in_vertex2 : tensor of shape [minibatch size, #vertex features] that
                                                               contains the vertices' features of the receptor protein.
                                                  in_edge1 : tensor that contains for each residue in the ligand protein
                                                             the edges' features of all its neighbors.
                                                  in_edge2 : tensor that contains for each residue in the receptor protein
                                                             the edges' features of all its neighbors.
                                                  in_hood_indices1 : tensor that contains for each residue in the ligand
                                                                     protein the index of its neighbors.
                                                  in_hood_indices1 : tensor that contains for each residue in the receptor
                                                                     protein the index of its neighbors.
                                                  examples : parings between the ligand and receptor residues.
                                                  labels : correct classification of their interaction.
                                                  dropout_keep_prob : probability with which the new values are the
                                                                      obtained one, scaled up by 1 / dropout_keep_prob.
                                                                      Otherwise the output is 0."""

    in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds, labels, dropout_keep_prob = model_variables_list
    feed_dict = {
        in_vertex1: minibatch["l_vertex"], in_edge1: minibatch["l_edge"],
        in_vertex2: minibatch["r_vertex"], in_edge2: minibatch["r_edge"],
        in_hood_indices1: minibatch["l_hood_indices"],
        in_hood_indices2: minibatch["r_hood_indices"],
        examples: minibatch["label"][:, :2],
        labels: minibatch["label"][:, 2],
        dropout_keep_prob: 0.5  # dropout_keep
    }
    return feed_dict



def build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size):
    """This function defines how the variable resulting from the architecture of the model are produced. The architecture
          is composed by the two "legs" (left, or 1, and right, or 2) of two convolutional layers (the left one for the ligand
          and the right one for the receptor proteins), the merging of their two outputs in a single tensor, which
          is then fed into two dense layers. The final dense layers associate to each pair's combination a single value.
          This tensor is than averaged, so that the resulting tensor has half of its dimension. At the end, to each pair of
          residues a single value, corresponding to the prediction value, is associated.
          A more in depth discussion can be again found at https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf
                     Parameters:
                         in_nv_dims : number of features of a vertex.
                         in_ne_dims : number of features of an edge.
                         in_nhood_size : number of neighbors.
                     Returns:
                         in_vertex1 : tensor with the features of each node considered from the ligand protein.
                         in_edge1 : tensor that contains for each residue in the ligand protein
                                    the edges' features of all its neighbors.
                         in_hood_indices1 : tensor with the index of the neighbors of each considered node in the
                                            ligand protein.
                         in_vertex2 : tensor with the features of each node considered from the receptor protein.
                         in_edge2 : tensor that contains for each residue in the receptor protein
                                    he edges' features of all its neighbors.
                         in_hood_indices2 : tensor with the index of the neighbors of each considered node in
                                            the receptor protein.
                         examples : pairings between the ligand and receptor proteins' residues.
                         preds : prediciton about the interaction or absence of it between the pairs.
                         labels : correct classification of the relation between the pairs.
                         dropout_keep_prob : probability with which the new values are the obtained one, scaled
                                             up by 1 / dropout_keep_prob. Otherwise the output is 0."""
    in_vertex1 = tf.placeholder(tf.float32, [None, in_nv_dims], "vertex1")
    in_vertex2 = tf.placeholder(tf.float32, [None, in_nv_dims], "vertex2")
    in_edge1 = tf.placeholder(tf.float32, [None, in_nhood_size, in_ne_dims], "edge1")
    in_edge2 = tf.placeholder(tf.float32, [None, in_nhood_size, in_ne_dims], "edge2")
    in_hood_indices1 = tf.placeholder(tf.int32, [None, in_nhood_size, 1], "hood_indices1")
    in_hood_indices2 = tf.placeholder(tf.int32, [None, in_nhood_size, 1], "hood_indices2")

    # Input from the ligand protein that is going to the left branch
    input1 = in_vertex1, in_edge1, in_hood_indices1
    # Input from the receptor protein that is going to the right branch
    input2 = in_vertex2, in_edge2, in_hood_indices2

    examples = tf.placeholder(tf.int32, [None, 2], "examples")
    labels = tf.placeholder(tf.float32, [None], "labels")
    dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

    layer_no = 1
    # First convolutional layer of the left branch
    name = "left_branch_{}_{}".format("node_average", layer_no)
    with tf.name_scope(name):
        output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
        input1 = output, in_edge1, in_hood_indices1

    # Firs convolutional layer of the right branch
    name = "right_branch_{}_{}".format("node_average", layer_no)
    with tf.name_scope(name):
        # the weights (params) are the ones of the left_branch
        output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
        input2 = output, in_edge2, in_hood_indices2

    layer_no = 2
    # Second convolutional layer of the left branch
    name = "left_branch_{}_{}".format("node_average", layer_no)
    with tf.name_scope(name):
        output, params = node_average_model(input1, None, filters=256, dropout_keep_prob=0.5)
        input1 = output, in_edge1, in_hood_indices1

    # Second convolutional layer of the right branch
    name = "right_branch_{}_{}".format("node_average", layer_no)
    with tf.name_scope(name):
        # the weights (params) are the ones of the left_branch
        output, _ = node_average_model(input2, params, filters=256, dropout_keep_prob=0.5)
        input2 = output, in_edge2, in_hood_indices2

    # The output of the two branches are merged
    layer_no = 3
    name = "{}_{}".format("merge", layer_no)
    input = input1[0], input2[0], examples
    with tf.name_scope(name):
        input = merge(input)

    # First dense layer
    layer_no = 4
    name = "{}_{}".format("dense", layer_no)
    with tf.name_scope(name):
        input = dense(input, out_dims=512, dropout_keep_prob=0.5, nonlin=True, trainable=True)

    # Second dense layer
    layer_no = 5
    name = "{}_{}".format("dense", layer_no)
    with tf.name_scope(name):
        input = dense(input, out_dims=1, dropout_keep_prob=0.5, nonlin=False, trainable=True)

    # Average layer
    layer_no = 6
    name = "{}_{}".format("average_predictions", layer_no)
    with tf.name_scope(name):
        preds = average_predictions(input)

    return [in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds, labels,
            dropout_keep_prob]
# Positive and negative examples ratio
pn_ratio = 0.1
# Parameter that determines the step size at each iteration while moving toward the minimum of the loss function.
learning_rate = 0.05

def loss_op(preds, labels):
    """This function defines the optimization algorithm. The loss is given by the cross entropy.
                    Parameters:
                        preds: the prediction about the existence of an interaction between pairs, given as output by the
                               final average layer.
                        labels : the correct labels that tell if a pair is interacting (label=+1) or not (label=-1).
                    Returns:
                        loss : cross entropy loss, determined by the difference between preds and labels."""
    with tf.name_scope("loss"):
        # Weights vector: negative examples are less relevant than the positive ones
        scale_vector = (pn_ratio * (labels - 1) / -2) + ((labels + 1) / 2)
        logits = tf.concat([-preds, preds], axis=1)
        # First column: negative labels are now positive and positive labels are nullified
        # Second column: positive label are preserved and negative labels are zeroed
        labels_stacked = tf.stack([(labels - 1) / -2, (labels + 1) / 2], axis=1)
        loss = tf.losses.softmax_cross_entropy(labels_stacked, logits, weights=scale_vector)
        return loss
