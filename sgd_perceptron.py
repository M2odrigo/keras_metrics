'''
train perceptron and plot the total loss in each epoch.

:param X: data samples
:param Y: data labels
:return: weight vector as a numpy array
'''
import numpy as np
import csv
import os
from matplotlib import pyplot as plt

def perceptron_plot(filename, layer, e, cant_input):
    dataset = np.loadtxt(filename, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:int(cant_input)]
    #add the bias term -1
    X = np.insert(X, int(cant_input),-1, axis=1)
    Y = dataset[:,int(cant_input)]
    Y[Y == 0] = -1
    w = np.zeros(len(X[0]))
    eta = 1
    n = 100
    errors = []
    mala_clasif = []

    for t in range(n):
        total_error = 0
        cont_error = 0
        cont_total_input=0
        for i, x in enumerate(X):
            cont_total_input=cont_total_input+1
            if (np.dot(X[i], w)*Y[i]) <= 0.0:
                total_error += (np.dot(X[i], w)*Y[i])
                #print(total_error)
                w = w + eta*X[i]*Y[i]
                cont_error = cont_error +1
        errors.append(total_error*-1)
        mala_clasif.append(cont_error)
        #print('cantidad de errores: ' + str(cont_error))

    #print(mala_clasif)
    #input('this_?')
    #fields=[layer, e, str(cont_total_input),np.amin(mala_clasif), mala_clasif]
    mn,idx = min( (mala_clasif[i],i) for i in range(len(mala_clasif)) )
    fields=[layer, e, str(cont_total_input),mn, idx]
    with open('hidden_perceptron_error.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    #plt.show()
