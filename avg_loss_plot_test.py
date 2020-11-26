import matplotlib.pyplot as plt
import numpy as np

#load the testing results of the two compared model 
original = "Testing_loss_noedge.txt"
modified = "Sigmoidal_Testing_loss_noedge.txt" #insert the training results of the desired model


# Import data as a list of numbers
with open(original) as textFile:
    loss,model,roc_auc = np.loadtxt(fname=textFile,delimiter=',',
                                    skiprows=1,
                                    unpack=True)
    model_sort = np.argsort(model)
    loss = loss[model_sort]
    model = model[model_sort]
    roc_auc = roc_auc[model_sort]

with open(modified) as textFile:
    loss1,model1,roc_auc1 = np.loadtxt(fname=textFile, delimiter=',',
                                        skiprows=1,
                                        unpack=True)
    model_sort1 = np.argsort(model1)
    loss1 = loss1[model_sort1]
    model1 = model1[model_sort1]
    roc_auc1 = roc_auc1[model_sort1]

#creation of figure and axes andle
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

#the plot legend is adapted to the selected model
if "Sigmoidal" in modified:
    original_label = 'Models with ReLU as transition function'
    modified_label = 'Models with tanh as transition function'
elif "IndPar" in modified:
    original_label = 'With the same weights for the two "legs"'
    modified_label = 'With independent weights for the two "legs"'
else:
    original_label = 'Without the edges\' features'
    modified_label = 'With the edges\' features'


ax1.plot(model, loss, '-o', label= original_label)
ax1.plot(model1, loss1, '-o', label= modified_label)
ax2.plot(model, roc_auc, '-o', label= original_label)
ax2.plot(model1, roc_auc1, '-o', label= modified_label)

ax2.set_xlabel('Model trained with epoch number')
ax2.set_ylabel('roc_auc')
ax1.set_ylabel('Average loss for testing')
ax1.legend()
ax2.legend()
plt.show()
