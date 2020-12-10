"""
The documentation about the functions build_graph_conv_model, loss_op and build_feed_dict
can be found at https://github.com/gretagrassmann/SoftwareAndComputingEXAM/blob/Customizable/graph_conv.py.
A more in depth explanation of the theory behind this model can be found at
https://raw.github.com/gretagrassmann/SoftwareAndComputingEXAM/master/GCN.pdf.
"""
import os
import pickle
import copy
from sklearn.metrics import roc_curve, auc
from graph_conv import *
import time
start_time = time.time()
if __name__=='__main__':
  """
  The following lines can be deleted if another set of data is used:
              DBD DATA BEGINNING
  """
  # The testing data are downloaded and unzipped
  url_test_data = 'https://raw.github.com/pchanda/Graph_convolution_with_proteins/master/data/test.cpkl.gz'
  file_name2 = re.split(pattern='/', string=url_test_data)[-1]
  r2 = request.urlretrieve(url=url_test_data, filename=file_name2)
  txt2 = re.split(pattern=r'\.', string=file_name2)[0] + ".txt"

  with gzip.open(file_name2, 'rb') as f_in2:
      with open(txt2, 'wb') as f_out2:
          shutil.copyfileobj(f_in2, f_out2)
  """             DBD DATA ENDING            """

  # The models corresponding to these number of epochs are going to be tested in a cycle
  n = [0,10,50,100,149]
  # Load the testing data
  test_data_file = os.path.join('test.txt')
  test_list, test_data = pickle.load(open(test_data_file, 'rb'), encoding='latin1')

  # Number of features of a vertex
  in_nv_dims = test_data[0]["l_vertex"].shape[-1]
  # Number of features of an edge
  in_ne_dims = test_data[0]["l_edge"].shape[-1]
  # Number of neighbors
  in_nhood_size = test_data[0]["l_hood_indices"].shape[1]

  # Defines the variables used in the model's architecture
  model_variables_list = build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size)
  in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob = model_variables_list

  loss = loss_op(preds, labels)

  saver = tf.train.Saver()
  with tf.Session() as sess:
     # Set up tensorflow session
     for model_num in n:
         """
         The model resulting from the training done with model_num epochs is retrieved and used to classify
         a new class of proteins to perform a testing.
         """
         saver.restore(sess, './saved_models/model_%d.ckpt' % (model_num))
         print(" Using model %d " % (model_num), " for testing %d proteins" % (len(test_data)))

         all_preds = []
         all_labels = []
         all_losses = []
         count = 0
         """
         The mean loss and the area under the ROC curve for the considered model are calculated and saved.
         """
         for prot_data in test_data:
           temp_data = copy.deepcopy(prot_data)
           # Number of labels for this protein molecule
           n = prot_data['label'].shape[0]
           count = count +1
           print("Protein number = %d" %count)
           print("Number of residues couples = %d" %n)
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
         print('test mean loss = ',np.mean(all_losses))
         print('test roc_auc = ',roc_auc)

         # The mean loss and the area under the ROC curve for each tested model are saved
         with open("Testing_loss.txt","a+") as f:
             if model_num == 0:
                 f.write('Average loss, model number, roc_auc \n')
             f.write(str(np.mean(all_losses))+','+ str(model_num)+','+str(roc_auc)+"\n")

print("My program took", time.time() - start_time, "to run")
