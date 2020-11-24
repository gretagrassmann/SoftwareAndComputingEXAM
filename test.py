import os
import pickle
import copy
from sklearn.metrics import roc_curve, auc
from graph_conv import *
import time
start_time = time.time()
if __name__=='__main__':

  n = [0,10,50,100,150] #The models corresponding to these number of epochs are going to be tested in a cycle
  #load the testing data
  test_data_file = os.path.join('test.txt')
  test_list, test_data = pickle.load(open(test_data_file, 'rb'), encoding='latin1')

  in_nv_dims = test_data[0]["l_vertex"].shape[-1]
  in_ne_dims = test_data[0]["l_edge"].shape[-1]
  in_nhood_size = test_data[0]["l_hood_indices"].shape[1]

  model_variables_list = build_graph_conv_model(in_nv_dims, in_ne_dims, in_nhood_size)
  in_vertex1, in_edge1, in_hood_indices1, in_vertex2, in_edge2, in_hood_indices2, examples, preds,labels, dropout_keep_prob = model_variables_list

  loss = loss_op(preds, labels)

  saver = tf.train.Saver()
  with tf.Session() as sess:
     # set up tensorflow session
     for model_num in n:
         saver.restore(sess, './Edge_saved_models/model_%d.ckpt' % (model_num))
         print(" Using model %d " % (model_num), " for testing %d proteins" % (len(test_data)))

     all_preds = []
     all_labels = []
     all_losses = []
     count = 0

     for prot_data in test_data:
       temp_data = copy.deepcopy(prot_data)
       n = prot_data['label'].shape[0] #no of labels for this protein molecule.
       count = count +1
       print("Protein number = %d" %count)
       print("Number of residues couples = %d" %n)
       #split the labels into chunks of minibatch_size.
       batch_split_points = np.arange(0,n,minibatch_size)[1:]
       batches = np.array_split(prot_data['label'],batch_split_points)
       for a_batch in batches:
         temp_data['label'] = a_batch
         feed_dict = build_feed_dict(model_variables_list, temp_data)
         res = sess.run([loss,preds,labels], feed_dict=feed_dict)
         pred_v = np.squeeze(res[1])
         if len(pred_v.shape)==0:
           pred_v = [pred_v]
           all_preds += pred_v
         else:
           pred_v = pred_v.tolist()
           all_preds += pred_v
         all_labels += res[2].tolist()
         all_losses += [res[0]]



     fpr, tpr, _ = roc_curve(all_labels, all_preds)
     roc_auc = auc(fpr, tpr)
     print('test mean loss = ',np.mean(all_losses))
     print('test roc_auc = ',roc_auc)

     with open("Edge_Testing_loss.txt","a+") as f:
         if model_num == 0:
             f.write('Average loss, model number, roc_auc \n')
         f.write(str(np.mean(all_losses))+','+ str(model_num)+str(roc_auc)+"\n")

print("My program took", time.time() - start_time, "to run")
