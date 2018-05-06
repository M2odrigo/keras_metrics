import numpy as np
from matplotlib import pyplot as plt

dataset = np.loadtxt('output_1.csv', delimiter=",")
cant_input=8
# split into input (X) and output (Y) variables
X = dataset[:,0:cant_input]
#add the bias term -1
X = np.insert(X, cant_input,-1, axis=1)
y = dataset[:,cant_input]
y[y == 0] = -1

def perceptron_sgd_plot(X, Y):
    '''
    train perceptron and plot the total loss in each epoch.
    
    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 4
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
    
    #print(errors)    
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()
    
    return w

print(perceptron_sgd_plot(X,y))
