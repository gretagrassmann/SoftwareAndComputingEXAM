import matplotlib.pyplot as plt
import numpy as np
original = "C:\\Users\\Cobal\\Desktop\\avg_loss_train.txt"
modified = "C:\\Users\\Cobal\\Desktop\\Edge_avg_loss_train.txt"

if "Sigmoidal" in modified:
    # Import data as a list of numbers
    with open(original) as textFile:
        data = textFile.read().split()          # split based on spaces
        data = [float(point) for point in data] # convert strings to floats

    with open(modified) as textFile:
        data1 = textFile.read().split()          # split based on spaces
        data1 = [float(point) for point in data1] # convert strings to floats

    x = np.arange(1,len(data)+1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, data, label='With ReLU as transition function')
    ax.plot(x, data1, label='With tanh as transition function')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Average loss')
    #plt.title('Comparison ')
    ax.legend()
    plt.show()

elif "IndPar" in modified:
    # Import data as a list of numbers
    with open(original) as textFile:
        data = textFile.read().split()  # split based on spaces
        data = [float(point) for point in data]  # convert strings to floats

    with open(modified) as textFile:
        data1 = textFile.read().split()  # split based on spaces
        data1 = [float(point) for point in data1]  # convert strings to floats

    x = np.arange(1, len(data) + 1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, data, label='With the same weights for the two "legs"')
    ax.plot(x, data1, label='With independent weights for the two "legs"')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Average loss')
    # plt.title('Comparison ')
    ax.legend()
    plt.show()

else:
    # Import data as a list of numbers
    with open(original) as textFile:
        data = textFile.read().split()  # split based on spaces
        data = [float(point) for point in data]  # convert strings to floats

    with open(modified) as textFile:
        data1 = textFile.read().split()  # split based on spaces
        data1 = [float(point) for point in data1]  # convert strings to floats

    x = np.arange(1, len(data) + 1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, data, label='Without the edges\' features')
    ax.plot(x, data1, label='With the edges\' features')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Average loss')
    # plt.title('Comparison ')
    ax.legend()
    plt.show()