import numpy as np
import csv
from matplotlib import pyplot as plt

dataset = np.loadtxt('hidden_0_activations.csv', delimiter=",")
cant_input=30
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
    eta = 0.5
    #n = 11
    n = 500
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
                w = w + eta*X[i]*Y[i]
                cont_error = cont_error +1
        errors.append(total_error*-1)
        mala_clasif.append(cont_error)
    
    #mn,idx = min((mala_clasif[i],i) for i in range(len(mala_clasif))) 
    mala_clasif.sort()
    print(mala_clasif)
    mn = mala_clasif[2]
    error_minimo = mn/cont_total_input
    print(" error minimo "+ str(mn) + ' acc. '  +str(100 -(error_minimo*100)) + ' total entradas: ' + str(cont_total_input))
    #print(errors)    
    #plt.plot(errors)
    #plt.xlabel('Epoch')
    #plt.ylabel('Total Loss')
    #plt.show()
    
    return w


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

    #obtenemos el error minimo entre los epoch y su ubicacion-indice
    #mn,idx = min((mala_clasif[i],i) for i in range(len(mala_clasif)))
    #ordenamos la lista y agarramos el segundo o tercer valor, el primero no cuenta porque todo es muy proximo a zero debido a la inicializacion de los pesos en zeros.
    mala_clasif.sort()
    print(mala_clasif)
    mn = mala_clasif[2]
    #vamos a calcular la metrica por el menor valor y por el utlimo valor de error en los epoch
    error_minimo = mn/cont_total_input
    error_ultimo_epoch = cont_error/cont_total_input
    print(mala_clasif)
    #cargamos los valores a escribir en el CSV
#se carga el error minimo del perceptron y el ultimo error en el ultimo epoch
    #fields=[layer, e, str(cont_total_input),mn, idx, error_minimo,error_ultimo_epoch]
#se carga solo el error minimo entre los epochs del percecptron
    fields=[layer, e, str(cont_total_input),mn, error_minimo,str(100 -(error_minimo*100))]
    with open('train_hidden_perceptron_error.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    return fields
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    #plt.show()

#print(perceptron_sgd_plot(X,y))
print(perceptron_plot('hidden_0_activations.csv', 0, 500, 30))
