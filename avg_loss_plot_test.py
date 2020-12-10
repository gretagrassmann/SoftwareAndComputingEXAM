import matplotlib.pyplot as plt
import numpy as np

#load the testing results of the model
original = "Testing_loss.txt"


# Import data as a list of numbers
with open(original) as textFile:
    loss,model,roc_auc = np.loadtxt(fname=textFile,delimiter=',',
                                    skiprows=1,
                                    unpack=True)
    model_sort = np.argsort(model)
    loss = loss[model_sort]
    model = model[model_sort]
    roc_auc = roc_auc[model_sort]


#creation of figure and axes andle
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(model, loss, '-o')
ax2.plot(model, roc_auc, '-o')

ax2.set_xlabel('Model trained with epoch number')
ax2.set_ylabel('roc_auc')
ax1.set_ylabel('Average loss for testing')

plt.show()
