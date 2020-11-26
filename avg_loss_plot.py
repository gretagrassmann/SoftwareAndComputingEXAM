import matplotlib.pyplot as plt
import numpy as np

#load the trainig results of the two compared model 
original = "avg_loss_train.txt"
modified = "Sigmoidal_avg_loss_train.txt" #insert the training results of the desired model

with open(original) as textFile:
    data = textFile.read().split()  # split based on spaces
    data = [float(point) for point in data]  # convert strings to floats

with open(modified) as textFile:
    data1 = textFile.read().split()  # split based on spaces
    data1 = [float(point) for point in data1]  # convert strings to floats

x = np.arange(1, len(data) + 1)

fig = plt.figure()
ax = plt.subplot(111)

#the plot legend is adapted to the selected model
if "Sigmoidal" in modified:
    original_label = 'With ReLU as transition function'
    modified_label = 'With tanh as transition function'
elif "IndPar" in modified:
    original_label = 'With the same weights for the two "legs"'
    modified_label = 'With independent weights for the two "legs"'
else:
    original_label = 'Without the edges\' features'
    modified_label = 'With the edges\' features'

ax.plot(x, data, label= original_label)
ax.plot(x, data1, label= modified_label)
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Average loss')
plt.title('Comparison between two models')
ax.legend()
plt.show()