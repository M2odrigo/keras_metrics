'''
train perceptron and plot the total loss in each epoch.

:param X: data samples
:param Y: data labels
:return: weight vector as a numpy array
'''
import numpy as np
from matplotlib import pyplot as plt

dataset = np.loadtxt("hidden_1_activations.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
Y[Y == 0] = -1
w = np.zeros(len(X[0]))
eta = 1
n = 30
errors = []

for t in range(n):
    total_error = 0
    for i, x in enumerate(X):
        if (np.dot(X[i], w)*Y[i]) <= 0.0:
            total_error += (np.dot(X[i], w)*Y[i])
            print(total_error)
            w = w + eta*X[i]*Y[i]
    errors.append(total_error*-1)

plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.show()
