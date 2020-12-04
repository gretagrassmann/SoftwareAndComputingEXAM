import graph_conv
import hypothesis
import pytest
import numpy
from hypothesis import strategies as st
from hypothesis import settings
from hypothesis import given
from hypothesis import assume
import tensorflow as tf

# Changes a tensor in a numpy array: handy when looking at the single elements
tf.enable_eager_execution()
def tensor_to_array(tensor1):
    return tensor1.numpy()

################################ TEST FOR INITIALIZER

@given(matrix_shape = st.lists(min_size=1, max_size=2, elements=st.integers(min_value=1,max_value=260)))
def test_initializer(matrix_shape):
    """
        Tests:
        -If the output weights matrices (or vector of bias) have the right shape.
        -If the tensor for which init="zero" is all zero-valued.
        -If the tensor for which init="he" has a different distribution.
    """
    zero_weights_matrix = graph_conv.initializer("zero",matrix_shape)
    uniform_weights_matrix = graph_conv.initializer("he",matrix_shape)
    assert zero_weights_matrix.get_shape() == uniform_weights_matrix.get_shape() == matrix_shape
    # Checking if the sum of the absolute vale of all the elements is equal to zero for the
    # zero-valued initialization and not null for the one initialized with tf_random_uniform
    abs_zero_weights_matrix = tf.abs(zero_weights_matrix)
    abs_uniform_weights_matrix = tf.abs(uniform_weights_matrix)
    assert tf.math.reduce_sum(abs_zero_weights_matrix, axis=None).numpy() == 0
    assert tf.math.reduce_sum(abs_uniform_weights_matrix, axis=None).numpy() != 0


################################ TEST FOR NODE_AVERAGE_MODEL

@given(minibatch_size=st.integers(min_value=1,max_value=129),
       n_vertex_features=st.integers(min_value=1,max_value=70),
       n_edge_features=st.integers(min_value=1,max_value=70),
       n_neighbors=st.integers(min_value=1,max_value=20),
       n_filters=st.integers(min_value=1,max_value=260)
       )
@settings(deadline=None)
def test_node_average_model(minibatch_size,n_vertex_features,n_edge_features,n_neighbors,n_filters):
    """
            Tests:
            -If the weights tensors (the weights Wc and Wn in the convolutional layers, referred to the features of the
             center nodes and of their neighbors, and the vector of bias again in the convolutional layers)
             have the right shapes. In particular the case in which node_average_model receive as input params=None is
             analyzed, since otherwise the params tensor (of which the right shape is here tested) is simply decomposed.
            -If the tensor with the new values associated for each filter to each node, resulting from the convolution,
             has the right shape.

        """
    with tf.Session() as sess:
        # Tensor with uniformly randomly generated values that simulate the features of the vertices
        vertices = tf.random.stateless_uniform([minibatch_size,n_vertex_features],seed=(2,3),minval=1,maxval=100,dtype=tf.float32,name="vertex")
        # Tensor with uniformly randomly generated values that simulate the features of the edges
        edges = tf.random.stateless_uniform([minibatch_size,n_neighbors,n_edge_features],seed=(4,3),minval=1,maxval=100,dtype=tf.float32,name="edges")
        # Tensor with uniformly randomly generated values that simulate the neighborhood of each node
        nh_indices = tf.random.stateless_uniform([minibatch_size,n_neighbors,1],seed=(2,1),minval=1,maxval=100,dtype=tf.int64,name="hood_indices")
        input = vertices, edges, nh_indices

        output, params = graph_conv.node_average_model(input, None, n_filters, dropout_keep_prob=0.5)

    assert params["Wn"].get_shape() == params["Wc"].get_shape() == (n_vertex_features, n_filters)
    assert params["b"].get_shape() == n_filters
    assert output.get_shape() == (minibatch_size, n_filters)
    

################################ TEST FOR DENSE

@given(minibatch_size=st.integers(min_value=1,max_value=129),
       initial_dimension=st.integers(min_value=1,max_value=250),
       output_dimension=st.integers(min_value=1,max_value=10),
       )
@settings(deadline=None)
def test_dense(minibatch_size,initial_dimension,output_dimension):
    """
                Tests:
                -If the output tensor has the expected shape, that is output_dimension values for each one of the
                 2 * minibatch_size nodes.

            """
    with tf.Session() as sess:
        # Tensor with uniformly randomly generated values that simulate the features of the vertices
        input = tf.random.stateless_uniform([2*minibatch_size, initial_dimension], seed=(2, 3), minval=1, maxval=100,
                                               dtype=tf.float32)
        output = graph_conv.dense(input, out_dims=output_dimension, dropout_keep_prob=1.0, nonlin=True, trainable=True)

    assert output.get_shape() == (2*minibatch_size, output_dimension)


################################ TEST FOR MERGE

@given(minibatch_size=st.integers(min_value=1,max_value=129),
       n_filters=st.integers(min_value=1, max_value=260),
       n_pairings=st.integers(min_value=1, max_value=129)
       )
@settings(deadline=None)
def test_merge(minibatch_size,n_filters,n_pairings):
    """
                Tests:
                - If the first "quarter" of the output of the merging layer corresponds to the tensor with the values
                  assigned to the interacting ligand residues. The same test has be done for all the other
                  "quarters", but reporting it would have been redundant.
            """
    # Each ligand residue can be paired with zero or one receptor residue, and viceversa
    assume(n_pairings < minibatch_size)

    with tf.Session() as sess:
        # Tensor that simulate the values assigned for each filter to each ligand residue after the convolutional layers
        ligand_input = tf.random.stateless_uniform([minibatch_size, n_filters], seed=(2, 3), minval=1, maxval=100,
                                                   dtype=tf.float32)
        # Tensor that simulate the values assigned for each filter to each receptor residue after the convolutional layers
        receptor_input = tf.random.stateless_uniform([minibatch_size, n_filters], seed=(3, 1), minval=1, maxval=100,
                                                   dtype=tf.float32)

        # Construction of a tensor that simulate the interaction between ligand and receptor residue. Each residue has
        # only one interaction, or no interaction at all
        residues_indices = tf.range(0,minibatch_size,dtype=tf.int64)
        # Tensors with random extraction, without repetition, of n_pairings residues that interact with the other protein
        ligand_pairings = tf.reshape(tf.random.shuffle(residues_indices, seed=1, name=None)[0:n_pairings], [n_pairings,1])
        receptor_pairings = tf.reshape(tf.random.shuffle(residues_indices, seed=3, name=None)[0:n_pairings], [n_pairings,1])
        pairings = tf.concat([ligand_pairings, receptor_pairings], axis=1)

        input = ligand_input, receptor_input, pairings
        merge = graph_conv.merge(input)

        # The ligand residues that interact with some of the receptor residues
        out1 = tf.gather(ligand_input, pairings[:, 0])

        # The first "quarter" of merge is considered
        column_indices = tf.range(0,merge.get_shape()[1] // 2,1)
        half_merge = tf.gather(merge, column_indices, axis=1)
        row_indices = tf.range(0,merge.get_shape()[0] // 2,1)
        quarter_merge = tf.gather(half_merge,row_indices,axis=0)

        # Comparison between the tensor of the interacting ligand residues and the first quarter of the output of merge
        check = tf.math.equal(out1,quarter_merge)

        assert sess.run(tf.reduce_all(check)) ==True


################################ TEST FOR AVERAGE_PREDICTIONS

@given(minibatch_size=st.integers(min_value=1,max_value=129))
@settings(deadline=None)
def test_average_predicitions(minibatch_size):
    """
                Tests:
                - If the same result can be achieved with a different path: now the input tensor is reshaped so that the
                  results for ligand-receptor and receptor-ligand pairs are divided. Than the specular pairs' values are
                  summed. This tensor should than have the same shape of the average_predictions output [minibatchsize,1]
                  and as value the double of the corresponding value in the average_predictions output.
            """
    # Tensor that simulates the value associate to each ligand-receptor and receptor-ligand pair
    input = tf.random.stateless_uniform([2*minibatch_size,1],seed=(2,3),minval=1,maxval=10,dtype=tf.float32)

    code_predictions = graph_conv.average_predictions(input)

    # Alternative path to average_prediciton
    reshaped_input = tf.reshape(input[0:minibatch_size,0],[minibatch_size,1]),\
                         tf.reshape(input[minibatch_size:2*minibatch_size,0],[minibatch_size,1])
    sum = tf.reduce_sum(reshaped_input, 0)
    # The sum of two values is the double of their average
    comparison = tf.math.divide(sum,code_predictions)
    # Comparison should be a tensor of shape [minibatch_size,1] in which all elements are equal to 2
    assert tf.math.reduce_sum(comparison, axis=None).numpy() == 2*minibatch_size


################################ TEST FOR BUILD_GRAPH_CONV_MODEL

@given(n_vertex_features=st.integers(min_value=1,max_value=70),
       n_edge_features=st.integers(min_value=1,max_value=70),
       n_neighbors=st.integers(min_value=1,max_value=20)
        )
@settings(deadline=None)
def test_build_graph_conv_model(n_vertex_features,n_edge_features,n_neighbors):
    """
                Tests:
                - If the shape of the tensor are the expected one. This is indirectly a test for the function 
                  build_feed_dict, which simply decompose the minibatch data in the tensor in 
                  model_variable_list: indeed with this test we verified that the tensor in which the minibatch 
                  is divided have the wanted dimension.
            """
    with tf.Session() as sess:
        model_variables_list = graph_conv.build_graph_conv_model(n_vertex_features, n_edge_features, n_neighbors)
        in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds, labels, dropout_keep_prob = model_variables_list

    assert in_vertex1.get_shape().as_list() == in_vertex2.get_shape().as_list() ==[None,n_vertex_features]
    assert in_edge1.get_shape().as_list() == in_edge2.get_shape().as_list() ==[None, n_neighbors,n_edge_features]
    assert in_hood_indices1.get_shape().as_list() == in_hood_indices2.get_shape().as_list() ==[None, n_neighbors,1]
    assert examples.get_shape().as_list() == [None,2]
    assert labels.get_shape().as_list() == [None]
    assert preds.get_shape().as_list() == [None,1]