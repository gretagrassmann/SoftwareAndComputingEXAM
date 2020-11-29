"""
The documentation about the functions build_graph_conv_model, loss_op and build_feed_dict
can be found at https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/TransitionFunction/graph_conv.py
A more in depth explanation of the theory behind this model can be found at
https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf
"""
import os
import pickle
import copy
from sklearn.metrics import roc_curve, auc
from graph_conv import *

if __name__=='__main__':

  # Load the training data
  train_data_file = os.path.join('train.txt')
  train_list, train_data = pickle.load(open(train_data_file, 'rb'), encoding='latin1')

  # Number of features of a vertex
  in_nv_dims = train_data[0]["l_vertex"].shape[-1]
  # Number of features of an edge
  in_ne_dims = train_data[0]["l_edge"].shape[-1]
  # Number of neighbors
  in_nhood_size = train_data[0]["l_hood_indices"].shape[1]

  # Defines the variables used in the model's architecture
  model_variables_list = build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size)
  in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob = model_variables_list

  loss = loss_op(preds, labels)
  # Defines an optimization algorithm that trains the model
  with tf.name_scope("optimizer"):
      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  saver = tf.train.Saver(max_to_keep=250)

  with tf.Session() as sess:
     # Set up tensorflow session
     sess.run(tf.initialize_all_variables())
     print("Training Model")

     # File in which the average loss for each number of epochs is saved
     f = open("Sigmoidal_avg_loss_train.txt", "a")
     for epoch in range(0, num_epochs):
       """
       Trains model for one pass through training data, one protein at a time.
       Each protein is split into minibatches of paired examples.
       Features for the entire protein is passed to model, but only a minibatch of examples are passed.
       """
       prot_perm = np.random.permutation(len(train_data))
       ii = 0
       nn = 0
       avg_loss = 0
       # Loop through each protein
       for protein in prot_perm:
          # Extract data just for this protein
          prot_data = train_data[protein]
          pair_examples = prot_data["label"]
          n  = len(pair_examples)
          shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
          # Loop through each minibatch
          for i in range(int(n / minibatch_size)):
             # Extract data for this minibatch
             index = int(i * minibatch_size)
             example_pairs = pair_examples[shuffle_indices[index: index + minibatch_size]]
             minibatch = {}
             for feature_type in prot_data:
                 if feature_type == "label":
                     minibatch["label"] = example_pairs
                 else:
                     minibatch[feature_type] = prot_data[feature_type]
             # Train the model
             feed_dict = build_feed_dict(model_variables_list, minibatch)
             _,loss_v = sess.run([train_op,loss], feed_dict=feed_dict)
             avg_loss += loss_v
             ii += 1
          nn += n
       print("Epoch_end =",epoch,", avg_loss = ",avg_loss/ii," nn = ",nn)
       # The parameters and the average loss for all pairs defined after each epoch are saved
       ckptfile = saver.save(sess, './sigmoidal_saved_models/model_%d.ckpt'%(epoch))
       s = str(avg_loss / ii)
       f.write(s + "\n")

     f.close()

     all_preds = []
     all_labels = []
     all_losses = []
     """
        The mean loss and the area under the ROC curve for the final model (corresponding
        to the maximum number of epochs) are calculated and saved.
        """
     # Loop through each protein
     for prot_data in train_data:
       temp_data = copy.deepcopy(prot_data)
       # Number of labels for this protein molecule
       n = prot_data['label'].shape[0]
       # Split the labels into chunks of minibatch_size.
       batch_split_points = np.arange(0,n,minibatch_size)[1:]
       batches = np.array_split(prot_data['label'],batch_split_points)
       for a_batch in batches:
          temp_data['label'] = a_batch
          # Implements the model
          feed_dict = build_feed_dict(model_variables_list, temp_data)
          res = sess.run([loss,preds,labels], feed_dict=feed_dict)
          # List with the classification predicition of each pair in the minibatch
          pred_v = np.squeeze(res[1])
          # Since it has to be summed to a list, pred_v must always be considered as a list
          if len(pred_v.shape)==0:
             pred_v = [pred_v]
             all_preds += pred_v
          else:
             pred_v = pred_v.tolist()
             all_preds += pred_v
          all_labels += res[2].tolist()
          all_losses += [res[0]]

     # Area under the ROC curve
     fpr, tpr, _ = roc_curve(all_labels, all_preds)
     roc_auc = auc(fpr, tpr)
     print('mean loss = ',np.mean(all_losses))
     print('roc_auc = ',roc_auc)