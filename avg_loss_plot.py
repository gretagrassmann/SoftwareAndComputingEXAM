import matplotlib.pyplot as plt
import numpy as np

#load the trainig results of the model
original = "avg_loss_train.txt"

with open(original) as textFile:
    data = textFile.read().split()  # split based on spaces
    data = [float(point) for point in data]  # convert strings to floats

x = np.arange(1, len(data) + 1)

fig = plt.figure()
ax = plt.subplot(111)

ax.plot(x, data)
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Average loss')
ax.legend()
plt.show()