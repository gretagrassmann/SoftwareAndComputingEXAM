import matplotlib.pyplot as plt
import numpy as np
#from defs import *


relu = "C:\\Users\\Cobal\\Desktop\\Testing_loss_noedge.txt"
tanh = "C:\\Users\\Cobal\\Desktop\\Sigmoidal_Testing_loss_noedge.txt"


# Import data as a list of numbers
with open(relu) as textFile:
    loss,model,roc_auc = np.loadtxt(fname=textFile,delimiter=',',
                                    skiprows=1,
                                    unpack=True)
    model_sort = np.argsort(model)
    loss = loss[model_sort]
    model = model[model_sort]
    roc_auc = roc_auc[model_sort]

with open(tanh) as textFile:
    loss1,model1,roc_auc1 = np.loadtxt(fname=textFile, delimiter=',',
                                        skiprows=1,
                                        unpack=True)
    model_sort1 = np.argsort(model1)
    loss1 = loss1[model_sort1]
    model1 = model1[model_sort1]
    roc_auc1 = roc_auc1[model_sort1]

#CREATE FIGURE AND AXES HANDLES
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

#PLOT THINGS ON CORRECT PLOTS
ax1.plot(model, loss, '-o', label='Models with a ReLU activation function')
ax1.plot(model1, loss1, '-o', label='Models with a tanh activation function')
ax2.plot(model, roc_auc, '-o', label='Models with a ReLU activation function')
ax2.plot(model1, roc_auc1, '-o', label='Models with a tanh activation function')

#SETTING OF ALL LABELS
ax2.set_xlabel('Model trained with epoch number')
ax2.set_ylabel('roc_auc')
ax1.set_ylabel('Average loss for testing')
fig.suptitle('Behaviour of loss and roc_auc with training process')
ax1.legend()
ax2.legend()
plt.show()
